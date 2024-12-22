"""
训练模型
"""
import os.path

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from Dataset import HumTempDataset
from Model import HumTempPredictModel
from matplotlib import pyplot as plt

# 自动判断合适设备
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# epoch次数
epochs = 300
# batch_size 大小
batch_size = 100
if __name__ == '__main__':
    # 得到训练集
    train_dataset = HumTempDataset()
    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 得到验证集,传入temp_scaler，hum_scaler，因为我们的测试集和验证集等都要和训练集使用相同的scaler
    valid_dataset = HumTempDataset(data_type="train", temp_scaler=train_dataset.temp_scaler,
                                   hum_scaler=train_dataset.hum_scaler)
    valid_dataloader: DataLoader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    model = HumTempPredictModel().to(device)
    if os.path.exists("./pt/model.pth"):
        model.load_state_dict(torch.load("./pt/model.pth"))
    loss = MSELoss().to(device)
    optim = Adam(params=model.parameters(), lr=0.00001)

    train_loss_values = []
    valid_loss_values = []

    for epoch in range(epochs):
        model.train()
        loss_val = 0.0
        for data in train_dataloader:
            X, y = data[0].type(torch.FloatTensor).to(device), data[1].type(torch.FloatTensor).to(device)
            predict = model.forward(X)
            loss_value = loss(predict, y)
            optim.zero_grad()  # 反向传播，记得要把上一次的梯度清0
            loss_value.backward()
            loss_val += loss_value.item()  # 统计损失值
            optim.step()
            # print(loss_value)
        loss_val = loss_val / len(train_dataloader)
        print("train loss: ",loss_val)
        train_loss_values.append(loss_val)
        # 验证
        loss_val = 0.0
        model.eval()
        for data in valid_dataloader:
            X, y = data[0].type(torch.FloatTensor).to(device), data[1].type(torch.FloatTensor).to(device)
            predict = model.forward(X)
            loss_value = loss(predict, y)
            loss_val += loss_value.item()  # 统计损失值
        loss_val = loss_val / len(valid_dataloader)
        valid_loss_values.append(loss_val)
        print("train loss: ", loss_val)

    # 可视化损失下降
    plt.figure(figsize=(5, 4))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss Curve')
    plt.plot(range(epochs), train_loss_values)
    # plt.savefig("./chart/train.png")

    # 验证可视化
    plt.figure(figsize=(5, 4))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Valid Loss Curve')
    plt.plot(range(epochs), valid_loss_values)
    # plt.savefig("./chart/train.png")

    torch.save(model.state_dict(), './pt/model.pth')
