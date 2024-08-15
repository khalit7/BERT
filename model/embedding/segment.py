import torch.nn as nn

class SegmentEmbedding(nn.Embedding):
    def __init__(self, num_segments, embed_size=512):
        super(SegmentEmbedding, self).__init__(num_segments, embed_size,padding_idx=0)