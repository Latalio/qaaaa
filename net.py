import torch.nn as nn


class MatchLSTM(nn.Module):

    def __init__(self,
                 embedding_passage,
                 embedding_question):
        super(MatchLSTM, self).__init__()
        self.P = embedding_passage
        self.Q = embedding_question
        self.PreprocessLSTM = nn.ModuleDict({
            'passage': nn.LSTM(),
            'question': nn.LSTM()
        })
        pass

    def forward(self, *input):
        pass