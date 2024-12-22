"""
    测试模型
"""

import torch
from Model import HumTempPredictModel

# 自动判断合适设备
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = HumTempPredictModel()
model.load_state_dict(torch.load("pt/model.pth"))
model.to(device)
model.eval()  # 测试模式