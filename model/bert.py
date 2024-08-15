import torch.nn as nn
from model.attention.multihead_attention import MultiHeadAttention
from embedding.bert import BERTEmbedding
from other_layers.add_and_norm import AddNorm
from other_layers.feed_forward import PositionWiseFeedFoward
from output_layers import MaskedLanguageModelTaskLayer, NextSentencePredictionTaskLayer

class BERT_layer(nn.Module):
    def __init__(self,d_model, num_heads, d_ff, dropout=0.1):
        super(BERT_layer, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.position_wise_feed_forward = PositionWiseFeedFoward(d_model, d_ff, dropout)
        self.add_norm_1 = AddNorm(d_model)
        self.add_norm_2 = AddNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output, attention_scores = self.attention(x, x, x, mask)
        attention_output = self.add_norm_1(x, attention_output)

        feed_forward_output = self.position_wise_feed_forward(attention_output)
        output = self.add_norm_2(attention_output, feed_forward_output)

        return self.dropout(output), attention_scores
    

class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers=12, num_segments=3, max_seq_len=512, dropout=0.1):
        super(BERT, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_segments = num_segments
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.embedding = BERTEmbedding(vocab_size, d_model, max_seq_len, num_segments)

        self.layers = nn.ModuleList([BERT_layer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, segment_label, mask):
        x = self.embedding(x, segment_label)

        for layer in self.layers:
            x, attention_scores = layer(x, mask)

        return x, attention_scores
    
class BERT_LM(nn.Module):
    def __init__(self,bert_model):
        super(BERT_LM, self).__init__()

        self.bert = bert_model
        self.mlm_layer = MaskedLanguageModelTaskLayer(bert_model)
        self.next_sentence_layer = NextSentencePredictionTaskLayer(bert_model)

    def forward(self, x, segment_label, mask):
        x, attention_scores = self.bert(x, segment_label, mask)

        return self.next_sentence_layer(x),self.mlm_layer(x), attention_scores
