import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import get_extended_attention_mask

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
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        # next, we need to produce multiple heads for the proj
        # this is done by splitting the hidden state to self.num_attention_heads, each of size self.attention_head_size
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
        # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask):
        # Calculate the attention scores
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        scores = scores + attention_mask
        attn_weights = nn.Softmax(dim=-1)(scores)
        attn_weights = self.dropout(attn_weights)

        context_layer = torch.matmul(attn_weights, value)
        context_layer = context_layer.transpose(1, 2).contiguous().view(
            context_layer.size(0), -1, self.all_head_size
        )
        return context_layer

    def forward(self, hidden_states, attention_mask):
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # multi-head attention
        self.self_attention = BertSelfAttention(config)
        # add-norm
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # feed forward
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # another add-norm
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):
        output = dense_layer(output)
        output = dropout(output)
        return ln_layer(input + output)

    def forward(self, hidden_states, attention_mask):
        # Self-attention
        attn_output = self.self_attention(hidden_states, attention_mask)
        attn_output = self.add_norm(hidden_states, attn_output, self.attention_dense, self.attention_dropout, self.attention_layer_norm)

        # Feed-forward
        interm_output = self.interm_af(self.interm_dense(attn_output))
        layer_output = self.add_norm(attn_output, interm_output, self.out_dense, self.out_dropout, self.out_layer_norm)
        return layer_output

class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Embedding layers for token, position, type, POS, and NER embeddings
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.pos_tag_embedding = nn.Embedding(config.num_pos_tags, config.hidden_size)
        self.ner_tag_embedding = nn.Embedding(config.num_ner_tags, config.hidden_size)
        
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is a constant, register to buffer
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

        # BERT encoder
        self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        # For [CLS] token
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        self.init_weights()

    def embed(self, input_ids, pos_tag_ids=None, ner_tag_ids=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        inputs_embeds = self.word_embedding(input_ids)
        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = self.pos_embedding(pos_ids)
        tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)

        # Get POS and NER embeddings if provided
        if pos_tag_ids is not None and ner_tag_ids is not None:
            pos_tag_embeds = self.pos_tag_embedding(pos_tag_ids)
            ner_tag_embeds = self.ner_tag_embedding(ner_tag_ids)
            hidden_states = inputs_embeds + pos_embeds + tk_type_embeds + pos_tag_embeds + ner_tag_embeds
        else:
            hidden_states = inputs_embeds + pos_embeds + tk_type_embeds

        hidden_states = self.embed_layer_norm(hidden_states)
        hidden_states = self.embed_dropout(hidden_states)
        return hidden_states

    def encode(self, hidden_states, attention_mask):
        extended_attention_mask = get_extended_attention_mask(attention_mask, self.dtype)
        for layer_module in self.bert_layers:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states

    def forward(self, input_ids, attention_mask, pos_tag_ids=None, ner_tag_ids=None):
        embedding_output = self.embed(input_ids=input_ids, pos_tag_ids=pos_tag_ids, ner_tag_ids=ner_tag_ids)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)
        return {"last_hidden_state": sequence_output, "pooler_output": first_tk}
