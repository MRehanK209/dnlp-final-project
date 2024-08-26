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
        # although it is a bit unusual, we empirically observe that it yields better performance
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        # next, we need to produce multiple heads for the proj
        # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
        # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask=None):
        """
        output: [bs, seq_len, hidden_state]
        """
        # Calculate attention scores: Q * K^T
        scores = torch.matmul(query, key.transpose(-1, -2))  # [bs, num_attention_heads, seq_len, seq_len]
        
        # Scale the scores by square root of attention head size for stability
        scores = scores / math.sqrt(self.attention_head_size)  # [bs, num_attention_heads, seq_len, seq_len]
        
        # Add attention mask (if provided) to scores
        if attention_mask is not None:
            scores = scores + attention_mask  # [bs, num_attention_heads, seq_len, seq_len]
        
        # Apply softmax to normalize attention scores across seq_len dimension
        attention_probs = nn.Softmax(dim=-1)(scores)  # [bs, num_attention_heads, seq_len, seq_len]
        #attention_probs = F.softmax(scores, dim=-1)
        # Apply dropout to attention probabilities
        attention_probs = self.dropout(attention_probs)# [bs, num_attention_heads, seq_len, seq_len]
        
        # Weighted sum of values according to attention probabilities: softmax(Q * K^T) * V
        context = torch.matmul(attention_probs, value)  # [bs, num_attention_heads, seq_len, attention_head_size]
        
        # Transpose and reshape to recover the original shape: [bs, seq_len, hidden_state]
        bs = context.size(0)  # Infer batch size from context tensor
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.all_head_size)
        
        return context

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [bs, seq_len, hidden_state]
        attention_mask: [bs, 1, 1, seq_len]
        output: [bs, seq_len, hidden_state]
        """
        # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
        # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        # calculate the multi-head attention
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # multi-head attention
        self.self_attention = BertSelfAttention(config)
        #self.cross_attention = BertCrossAttention(config)
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
        """
        Apply residual connection to any layer and normalize the output.
        This function is applied after the multi-head attention layer or the feed forward layer.

        input: the input of the previous layer
        output: the output of the previous layer
        dense_layer: used to transform the output
        dropout: the dropout to be applied
        ln_layer: the layer norm to be applied
        """
         # Apply the dense layer to the output
        transformed_output = dense_layer(output)
        # Apply dropout
        dropped_output = dropout(transformed_output)
        # Add the original input back to the output (residual connection)
        residual_output = input + dropped_output
        # Apply layer normalization
        normalized_output = ln_layer(residual_output)
        return normalized_output

    def forward(self, hidden_states, attention_mask, cross_attention_mask=None):
        """
        A single pass of the bert layer.

        hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
        as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf.
        attention_mask: the mask for the attention layer

        each block consists of
        1. a multi-head attention layer (BertSelfAttention)
        2. a add-norm that takes the input and output of the multi-head attention layer
        3. a feed forward layer
        4. a add-norm that takes the input and output of the feed forward layer
        """
        # Step 1: Multi-Head Self Attention
        self_attention_output = self.self_attention(hidden_states, attention_mask)
        # Step 1.5: Multi-Head Cross Attention
        #self_cross_attention_output = self.cross_attention(self_attention_output,hidden_states, attention_mask)
         #Step 2: Apply Add-Norm after Multi-Head Attention
        attention_output = self.add_norm(hidden_states, self_attention_output, self.attention_dense,
                                        self.attention_dropout, self.attention_layer_norm)
        #attention_output = self.attention_dense(self_cross_attention_output)
        #attention_output = self.attention_layer_norm(attention_output + hidden_states)
        #attention_output = self.attention_dropout(attention_output)
        # Step 3: Feed Forward Layer
        intermediate_output = self.interm_af(self.interm_dense(attention_output))
        
        # Step 4: Apply Add-Norm after Feed Forward Layer
        #layer_output = self.add_norm(attention_output, intermediate_output, self.out_dense,
                                    #self.out_dropout, self.out_layer_norm)
        # Add-norm
        output = self.out_dense(intermediate_output)
        output = self.out_layer_norm(output + attention_output)
        layer_output = self.out_dropout(output)
        return layer_output


class BertModel(BertPreTrainedModel):
    """
    the bert model returns the final embeddings for each token in a sentence
    it consists
    1. embedding (used in self.embed)
    2. a stack of n bert layers (used in self.encode)
    3. a linear transformation layer for [CLS] token (used in self.forward, as given)
    """


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
        # position_ids (1, len position emb) is a constant, register to buffer
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

        # Get word embedding from self.word_embedding into input_embeds.
        inputs_embeds = self.word_embedding(input_ids)

        # Get position index and position embedding from self.pos_embedding into pos_embeds.
        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = self.pos_embedding(pos_ids)

        # Get token type ids, since we are not considering token type,
        # this is just a placeholder.
        tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)

        # Add three embeddings together
        embeddings = inputs_embeds + pos_embeds + tk_type_embeds

        # Apply embed_layer_norm and dropout
        embeddings = self.embed_layer_norm(embeddings)
        embeddings = self.embed_dropout(embeddings)

        # Return the hidden states.
        return embeddings


    def encode(self, hidden_states, attention_mask):
        """
        hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # get the extended attention mask for self attention
        # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
        # non-padding tokens with 0 and padding tokens with a large negative number
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(
            attention_mask, self.dtype
        )

        # pass the hidden states through the encoder layers
        for i, layer_module in enumerate(self.bert_layers):
            # feed the encoding from the last bert_layer to the next
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        return hidden_states


    def forward(self, input_ids, attention_mask):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # get the embedding for each input token
        embedding_output = self.embed(input_ids=input_ids)
        # feed to a transformer (a stack of BertLayers)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

        # get cls token hidden state
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)

        return {"last_hidden_state": sequence_output, "pooler_output": first_tk}

