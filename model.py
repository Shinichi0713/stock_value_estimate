

import torch
import torch.nn as nn
import numpy as np
import os
import common_resource

class StockPricePredictor(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=common_resource.hidden_size, output_size=1, num_layers=2):
        super(StockPricePredictor, self).__init__()

        # LSTMレイヤー
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)

        # 線形レイヤー
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]