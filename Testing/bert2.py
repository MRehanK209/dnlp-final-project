import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_bert import BertPreTrainedModel
from utils import get_extended_attention_mask, apply_custom_attention_mask


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # initialize the linear transformation layers for key, value, query
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # this dropout is applied to normalized attention scores following the original implementation of transformer
        # although it is a bit unusual, we empirically observe that it yields better performance
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask=None):
        """
        output: [bs, seq_len, hidden_state]
        """
        scores = torch.matmul(query, key.transpose(-1, -2))  # [bs, num_attention_heads, seq_len, seq_len]
        scores = scores / math.sqrt(self.attention_head_size)  # [bs, num_attention_heads, seq_len, seq_len]
        
        # Apply custom attention mask if provided
        if attention_mask is not None:
            scores = apply_custom_attention_mask(scores, attention_mask)

        attention_probs = nn.Softmax(dim=-1)(scores)  # [bs, num_attention_heads, seq_len, seq_len]
        attention_probs = self.dropout(attention_probs)# [bs, num_attention_heads, seq_len, seq_len]
        
        context = torch.matmul(attention_probs, value)  # [bs, num_attention_heads, seq_len, attention_head_size]
        
        bs = context.size(0)
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.all_head_size)
        
        return context

    def forward(self, hidden_states, attention_mask):
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value


class BertCrossAttention(BertSelfAttention):
    def forward(self, hidden_states, context_states, attention_mask, cross_attention_mask):
        # Transform hidden states (query) and context states (key, value)
        query_layer = self.transform(hidden_states, self.query)
        key_layer = self.transform(context_states, self.key)
        value_layer = self.transform(context_states, self.value)

        # Apply cross-attention with an additional cross_attention_mask
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        if cross_attention_mask is not None:
            attn_value = self.attention(key_layer, query_layer, value_layer, cross_attention_mask)

        return attn_value


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # multi-head attention
        self.self_attention = BertSelfAttention(config)
        self.cross_attention = BertCrossAttention(config) if config.add_cross_attention else None
        
        # add-norm
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # feed forward
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation_function = self.get_activation_function(config.hidden_act)
        
        # another add-norm
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def get_activation_function(self, act):
        if act == "gelu":
            return F.gelu
        elif act == "relu":
            return F.relu
        elif act == "swish":
            return F.silu  # SiLU is also known as Swish
        else:
            raise ValueError(f"Unsupported activation function: {act}")

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):
        transformed_output = dense_layer(output)
        dropped_output = dropout(transformed_output)
        residual_output = input + dropped_output
        normalized_output = ln_layer(residual_output)
        return normalized_output

    def forward(self, hidden_states, attention_mask, context_states=None, cross_attention_mask=None):
        # Multi-Head Self Attention
        self_attention_output = self.self_attention(hidden_states, attention_mask)
        attention_output = self.add_norm(hidden_states, self_attention_output, self.attention_dense,
                                         self.attention_dropout, self.attention_layer_norm)

        # Optional Cross-Attention
        if self.cross_attention and context_states is not None:
            cross_attention_output = self.cross_attention(attention_output, context_states, attention_mask, cross_attention_mask)
            attention_output = self.add_norm(attention_output, cross_attention_output, self.attention_dense,
                                             self.attention_dropout, self.attention_layer_norm)

        # Feed Forward Layer
        intermediate_output = self.activation_function(self.interm_dense(attention_output))
        
        # Add-Norm after Feed Forward Layer
        layer_output = self.add_norm(attention_output, intermediate_output, self.out_dense,
                                     self.out_dropout, self.out_layer_norm)
        return layer_output


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # embedding
        self.word_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

        # bert encoder
        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # for [CLS] token
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        self.init_weights()

    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        inputs_embeds = self.word_embedding(input_ids)

        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = self.pos_embedding(pos_ids)

        tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)

        embeddings = inputs_embeds + pos_embeds + tk_type_embeds

        embeddings = self.embed_layer_norm(embeddings)
        embeddings = self.embed_dropout(embeddings)

        return embeddings

    def encode(self, hidden_states, attention_mask, context_states=None, cross_attention_mask=None):
        extended_attention_mask = get_extended_attention_mask(attention_mask, self.dtype)

        for i, layer_module in enumerate(self.bert_layers):
            hidden_states = layer_module(hidden_states, extended_attention_mask, context_states, cross_attention_mask)

        return hidden_states

    def forward(self, input_ids, attention_mask, context_states=None, cross_attention_mask=None):
        embedding_output = self.embed(input_ids=input_ids)

        sequence_output = self.encode(embedding_output, attention_mask=attention_mask,
                                      context_states=context_states, cross_attention_mask=cross_attention_mask)

        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)

        return {"last_hidden_state": sequence_output, "pooler_output": first_tk}
