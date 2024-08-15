import torch.nn as nn
import torch

import math

class Attention(nn.Module):

    def __init__(self) -> None:
        super().__init__()


    def forward(self,query,key,value,mask=None):
        d_k = query.size(-1)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = nn.functional.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_scores, value)

        return output, attention_scores