import torch.nn as nn

class MaskedLanguageModelTaskLayer(nn.Module):
    def __init__(self, bert_model,):
        super(MaskedLanguageModelTaskLayer, self).__init__()
        self.bert_model = bert_model

        self.linear = nn.Linear(self.bert_model.d_model, self.bert_model.vocab_size)

    def forward(self, x):
        return nn.functional.softmax( self.linear(x),dim=-1)
    
class NextSentencePredictionTaskLayer(nn.Module):
    def __init__(self, bert_model):
        super(NextSentencePredictionTaskLayer, self).__init__()
        self.bert_model = bert_model

        self.linear = nn.Linear(self.bert_model.d_model, 2)

    def forward(self, x):
        return nn.functional.softmax( self.linear(x),dim=-1)