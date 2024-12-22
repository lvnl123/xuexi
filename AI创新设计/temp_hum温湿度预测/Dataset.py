"""
温湿度数据集
读取数据集，并利用MinMaxScaler对数据进行初步预处理
"""
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class HumTempDataset(Dataset):

    def __init__(self, time_length=10, data_type: str = 'train', temp_scaler: StandardScaler = None,
                 hum_scaler: StandardScaler = None):
        # 判断用户选择的数据类型是否是训练集、测试集、验证集三者之一。
        if data_type not in ["train", "test", "train"]:
            raise ValueError("类型只允许train test valid三种")
        # 同时读取温湿度数据集
        with open("./datasets/{}/temp_data.txt".format(data_type), "r") as f:
            temp_data = f.read().strip()
            temp_data_lines = temp_data.split("\n")
            temp_data = []
            for temp_line in temp_data_lines:
                temp_data = temp_data + temp_line.strip().split(" ")
            combine_temp_data = np.array(temp_data)
        # temp_data = np.array(temp_data)
        with open("./datasets/{}/hum_data.txt".format(data_type), "r") as f:
            hum_data = f.read().strip()
            hum_data_lines = hum_data.split("\n")
            hum_data = []
            for hum_line in hum_data_lines:
                hum_data = hum_data + hum_line.strip().split(" ")
            combine_hum_data = np.array(hum_data)
        combine_temp_data = np.array(combine_temp_data).reshape((-1, 1))
        combine_hum_data = np.array(combine_hum_data).reshape((-1, 1))

        # 最大最小归一化,只计算训练集的scaler，验证集和测试集都使用训练集的scaler
        if data_type == 'train':
            self.temp_scaler = StandardScaler()
            self.temp_scaler.fit(combine_temp_data)
            self.hum_scaler = StandardScaler()
            self.hum_scaler.fit(combine_hum_data)
        else:
            if temp_scaler is None or hum_scaler is None:
                raise ValueError("非train类型，需要提供temp_scaler和hum_scaler")
            self.temp_scaler = temp_scaler
            self.hum_scaler = hum_scaler

        # 定义data和label，指定X和Y
        datas = []
        labels = []
        # 遍历每一行数据集
        for temp_data,hum_data in zip(temp_data_lines, hum_data_lines):
            temp_data = np.array(temp_data.strip().split(" "),np.float32).reshape((-1, 1))
            hum_data = np.array(hum_data.strip().split(" "),np.float32).reshape((-1, 1))
            if len(temp_data) < time_length or len(hum_data) < time_length:
                raise ValueError("输入的时间长度大于数据集数据总长度！")
            if len(hum_data) != len(temp_data):
                raise ValueError("温湿度数据数量不一致！")
            for i in range(time_length, len(temp_data)):
                datas.append(np.array([self.temp_scaler.transform(temp_data[i - time_length:i]),
                                       self.hum_scaler.transform(hum_data[i - time_length:i])]))
                labels.append(np.array([self.temp_scaler.transform(np.array(temp_data[i]).reshape((-1, 1))),
                                        self.hum_scaler.transform(np.array(hum_data[i]).reshape((-1, 1)))]))
        # np.squeeze 去除多余的1维维度
        self.datas = np.squeeze(np.array(datas))
        self.labels = np.squeeze(np.array(labels))

        # 此时,datas的维度为len,param_count,time_length,即len为数据集长度，param_count为参数个数，分别为温度和湿度，因此为2，time_length选用的时间序列的长度。

        # print(self.datas.shape, self.datas)
        # 此时labels的维度为len,param_count，即通过time_length条数据预测下一时刻的结果。
        # print(self.labels.shape, self.labels)

    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

    def __len__(self):
        return len(self.datas)


if __name__ == '__main__':
    dataset = HumTempDataset()