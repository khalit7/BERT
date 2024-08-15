import torch.nn as nn
import torch

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_len, d_model) # pe shape: [max_len, d_model]

        position = torch.arange(0, max_len).unsqueeze(1).float() # position shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)) # div_term shape: [d_model//2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe shape: [max_len, d_model]
        pe = pe.unsqueeze(0)
        # pe shape: [1, max_len, d_model]
        
        self.register_buffer('pe', pe) # register pe as buffer will make pe to be saved in state_dict, moved to device when model is moved to device, but not trained. see https://stackoverflow.com/questions/57540745/what-is-the-difference-between-register-parameter-and-register-buffer-in-pytorch

        def forward(self, x):
            # x shape: [batch_size, seq_len, d_model]
            return self.pe[:, :x.size(1), :]