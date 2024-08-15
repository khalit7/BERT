import torch.nn as nn

from .positional import PositionalEmbedding
from .token import TokenEmbedding
from .segment import SegmentEmbedding

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_seq_len=512, num_segments=3):
        super(BERTEmbedding, self).__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.segment = SegmentEmbedding(num_segments=num_segments, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=max_seq_len)

    def forward(self, x, segment_label):
        x = self.token(x) + self.position(x) + self.segment(segment_label)
        return x