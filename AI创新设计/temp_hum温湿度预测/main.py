"""
    测试模型
"""
import os.path
import numpy as np
import torch
import paho.mqtt.client as mqtt

from Dataset import HumTempDataset
from Model import HumTempPredictModel


# 自动判断合适设备
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 得到训练集
train_dataset = HumTempDataset()

model = HumTempPredictModel()
model.load_state_dict(torch.load("pt/model.pth"))
model.to(device)
model.eval()  # 测试模式

# 定义mqtt
# MQTT服务器地址
broker_address = "192.168.1.2"  # 替换为你的MQTT代理地址
port = 1883  # MQTT代理的端口，默认为1883
client_id = "python_mqtt_client"  # MQTT客户端的唯一ID

# 定义数据缓存，缓存到指定个数才能进行计算
cache_data = []
time_length = 10  # 定义缓存个数


# 当连接到MQTT代理时调用的回调函数
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected with result code {reason_code}")
    # 订阅主题
    client.subscribe("base/temp_hum")


# 当接收到MQTT消息时调用的回调函数
def on_message(client, userdata, msg):
    temp, hum = msg.payload.decode('utf-8').split(",")
    print("当前温湿度：", temp, hum)
    # 输入数据做scaler
    temp = train_dataset.temp_scaler.transform(np.array([float(temp)]).reshape((-1, 1)))
    hum = train_dataset.hum_scaler.transform(np.array([float(hum)]).reshape((-1, 1)))
    if len(cache_data) == time_length:
        cache_data.remove(cache_data[0])
        cache_data.append([float(temp[0][0]), float(hum[0][0])])
        # print(cache_data)
        # batch = 1
        input_data = np.array(cache_data, dtype=np.float32).reshape((1, 2, time_length))
        input_data = torch.from_numpy(input_data).to(device)
        # 开始预测未来数据
        predict = model(input_data)
        # 当前获得predict是经过scale后的，因此需要转换为原始的结果
        predict = predict.detach().cpu().numpy()
        # print(predict)
        pred_temp = train_dataset.temp_scaler.inverse_transform(np.array([predict[0]]).reshape((-1, 1)))
        pred_hum = train_dataset.hum_scaler.inverse_transform(np.array([predict[1]]).reshape((-1, 1)))

        client.publish("predict/temp", "{}".format(pred_temp[0][0]))
        client.publish("predict/hum", "{}".format(pred_hum[0][0]))
        print("预测温湿度：", pred_temp[0][0], pred_hum[0][0])

    else:
        # print(cache_data)

        cache_data.append([float(temp[0][0]), float(hum[0][0])])


# 创建MQTT客户端实例
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

# 为客户端设置回调函数
client.on_connect = on_connect
client.on_message = on_message

# 连接到MQTT代理
client.connect(broker_address, port)
if __name__ == '__main__':
    # 将预测结果发送给MQTT,实时显示结果
    # 开始网络循环，处理接收到的消息和重连
    client.loop_forever()