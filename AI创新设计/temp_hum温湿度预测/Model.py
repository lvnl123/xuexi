"""
温湿度预测模型
模型结构：LSTM+FCN
"""
import torch
import torch.nn as nn

class HumTempPredictModel(nn.Module):
    def __init__(self):
        super(HumTempPredictModel, self).__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=64, num_layers=2, batch_first=True)
        self.fcn = nn.Sequential(
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=256),
            # nn.Dropout(p=0.1),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.LayerNorm(64),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        out = torch.squeeze(self.fcn(lstm_out))
        return out