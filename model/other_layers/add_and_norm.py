import torch.nn as nn

class AddNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, residual, out):
        return self.norm(residual + out)