"""
Description: BiGRU Model
Author: Sudao HE
Date: 2023/04/25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Neural Networks model : Bidirection GRU
"""


class BiGRU(nn.Module):

    def __init__(self, featurizer_hidden_dim, feature_length, dropout, device, bidirectional=True):
        super(BiGRU, self).__init__()
        self.hidden_dim_list = featurizer_hidden_dim
        self.feature_length = feature_length
        self.hidden_dim = self.hidden_dim_list[-1]
        self.features = nn.ModuleList()
        self.device = device
        gru_feature_length = self.feature_length
        for i in range(len(self.hidden_dim_list)):
            self.features.append(nn.GRU(gru_feature_length, self.hidden_dim_list[i], dropout=dropout, num_layers=1,
                            bidirectional=bidirectional))
            if bidirectional:
                gru_feature_length = self.hidden_dim_list[i] * 2
            else:
                gru_feature_length = self.hidden_dim_list[i]
        # linear
        if bidirectional:
            self.hidden2hidden = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4)
        else:
            self.hidden2hidden = nn.Linear(self.hidden_dim, self.hidden_dim * 4)
        self.hidden2label = nn.Linear(self.hidden_dim * 4, 3)
        #  dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [Batch, seq_num, feature_length]
        Return:
            y: [Batch, 3]
        """
        # x = x.view(-1, int(x.shape[2] // self.feature_length), self.feature_length)
        gru_out = torch.transpose(x, 0, 1)
        # print(gru_out.shape)
        for featurizer in self.features:
            gru_out, _ = featurizer(gru_out)
            # print(featurizer)
            # print(gru_out.shape)
        gru_out = torch.transpose(gru_out, 0, 1)
        gru_out = torch.transpose(gru_out, 1, 2)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        gru_out = F.tanh(gru_out)
        # linear
        hidden = self.hidden2hidden(gru_out)
        y = self.hidden2label(hidden)
        return y
