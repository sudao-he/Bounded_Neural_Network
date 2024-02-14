"""
Description: Deep CNN Model
Author: Sudao HE
Date: 2023/04/25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Neural Networks model : Deep CNN
"""


class DCNN(nn.Module):

    def __init__(self, feature_length, dropout, device, channels=[8, 16, 32, 64], kernel=5):
        super(DCNN, self).__init__()
        self.feature_length = feature_length
        self.features = nn.ModuleList()
        self.device = device
        channel = 1
        out_dim = feature_length
        for i in range(len(channels)):
            self.features.append(
                nn.Conv1d(in_channels=channel, out_channels=channels[i], kernel_size=kernel, padding=kernel//2))
            self.features.append(
                nn.ReLU())
            self.features.append(
                nn.MaxPool1d(4))
            channel = channels[i]
            out_dim = out_dim // 4
        # linear
        self.hidden2hidden = nn.Linear(channel*out_dim, channel*out_dim * 4)
        self.hidden2label = nn.Linear(channel*out_dim * 4, 3)
        #  dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [Batch, channels=1, feature_length]
        Return:
            y: [Batch, 3]
        """
        cnn_out = x
        print(cnn_out.shape)
        for featurizer in self.features:
            cnn_out = featurizer(cnn_out)
            print(featurizer)
            print(cnn_out.shape)
        cnn_out = cnn_out.view(cnn_out.shape[0], -1)
        # linear
        hidden = self.hidden2hidden(cnn_out)
        y = self.hidden2label(hidden)
        return y
