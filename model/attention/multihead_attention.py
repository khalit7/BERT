import torch.nn as nn

from .attention import Attention

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, h):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.h = h # number of heads
        self.d_k = d_model // h # head dimentions
        
        '''
        we perform three linear tranformations for query, key and value.
        instead of tranforming each one to tensor of shape d_model*d_k h times,
        we can tranform each one to tensor of shape d_model*d_model once and then split it to h heads.
        '''
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        self.output_linear = nn.Linear(d_model, d_model)

        self.attention = Attention()
        
    def forward(self, query, key, value, mask=None):
        # query, key, value shape: [batch_size, seq_len, d_model]
        batch_size = query.size(0)
        
        # Linear transformations
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        # Reshape tensors from [batch_size, seq_len, d_model] to [batch_size, num_heads, seq_len, d_k]
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_ouput,attention_scores = self.attention(query, key, value, mask=mask)
        # attention_output shape: [batch_size, num_heads, seq_len, d_k]
        # attention_scores shape: [batch_size, num_heads, seq_len, seq_len]

        # Reshape and concatenate heads
        attention_ouput = attention_ouput.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # attention_output shape: [batch_size, seq_len, d_model]

        # Linear transformation for output
        multihead_output = self.output_linear(attention_ouput)
        # multihead_output shape: [batch_size, seq_len, d_model]
        
        return multihead_output, attention_scores