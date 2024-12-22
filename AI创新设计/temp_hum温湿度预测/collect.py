"""
数据采集与收集
MQTT接收传感器数据
"""

import paho.mqtt.client as mqtt
import os

# MQTT服务器地址
broker_address = "192.168.1.2"  # 替换为你的MQTT代理地址
port = 1883  # MQTT代理的端口，默认为1883
client_id = "python_mqtt_client"  # MQTT客户端的唯一ID，不冲突即可

# 收集 train train test数据集
dataset_type = "train"

hum_data_url = "./datasets/{}/hum_data.txt".format(dataset_type)
temp_data_url = "./datasets/{}/temp_data.txt".format(dataset_type)

hum_data_file = open(hum_data_url, "a+")
temp_data_file = open(temp_data_url, "a+")

if os.path.exists(hum_data_url):
    with open(hum_data_url, "r") as f:
        text = f.read()
        if text != "" and "\n" != text[-1]:
            # 判断末尾是否为\n，如果不是则添加一个\n
            hum_data_file.write("\n")
            hum_data_file.flush()

if os.path.exists(temp_data_url):
    with open(temp_data_url, "r") as f:
        text = f.read()
        if text != "" and "\n" != text[-1]:
            # 判断末尾是否为\n，如果不是则添加一个\n
            temp_data_file.write("\n")
            temp_data_file.flush()


# 当连接到MQTT代理时调用的回调函数
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected with result code {reason_code}")
    # 订阅主题
    client.subscribe("base/temp_hum")


# 当接收到MQTT消息时调用的回调函数
def on_message(client, userdata, msg):
    temp, hum = msg.payload.decode('utf-8').split(",")
    print(temp, hum)
    # print(f"接收到消息： '{msg.payload.decode('utf-8')}',主题： '{msg.topic}'")
    hum_data_file.write(f"{hum} ")
    temp_data_file.write(f"{temp} ")
    hum_data_file.flush()
    temp_data_file.flush()


# 创建MQTT客户端实例
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

# 为客户端设置回调函数
client.on_connect = on_connect
client.on_message = on_message

# 连接到MQTT代理
client.connect(broker_address, port)

# 开始网络循环，处理接收到的消息和重连
client.loop_forever()