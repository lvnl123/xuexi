import socket
from machine import Pin, I2S
import network


# 初始化I2S对象
def init_i2s():
    audio_in = I2S(
        0,  # I2S编号
        sck=Pin(26, Pin.OUT),  # 串行时钟
        ws=Pin(25, Pin.OUT),  # 左右选择（字选择）
        sd=Pin(22, Pin.IN),  # 数据输入
        mode=I2S.RX,  # 接收模式
        bits=16,  # 每个样本的位数: 16
        format=I2S.MONO,  # 单声道格式
        rate=16000,  # 采样率，单位：Hz
        ibuf=4096  # 缓冲区大小，单位：字节
    )
    return audio_in


# 录音函数
def record_audio(total_samples, filename="audio_data.pcm"):
    audio_in = init_i2s()
    read_buffer = bytearray(4096)
    leftover_buffer = bytearray()

    with open(filename, 'wb') as f:
        samples_read = 0
        while samples_read < total_samples:
            num_read = audio_in.readinto(read_buffer)
            if num_read > 0:
                combined_buffer = leftover_buffer + read_buffer[:num_read]
                sample_bytes = len(combined_buffer) // 2 * 2  # 确保是样本大小的整数倍
                f.write(combined_buffer[:sample_bytes])
                samples_read += sample_bytes // 2  # 每个样本16位（2字节）
                leftover_buffer = combined_buffer[sample_bytes:]
    audio_in.deinit()


# 分块发送文件到PC端
def send_file_to_pc(filename="audio_data.pcm", host="192.168.127.62", port=12345, chunk_size=4096):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        print(f"正在连接到 {host}:{port}...")
        sock.connect((host, port))
        print("连接成功，开始发送文件...")

        with open(filename, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sock.sendall(chunk)
        print("文件发送完成")

    except Exception as e:
        print(f"发送文件时发生错误: {e}")

    finally:
        sock.close()


# WiFi 配置
def connect_wifi(ssid, password):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print("正在连接 WiFi...")
        wlan.connect(ssid, password)
        while not wlan.isconnected():
            pass
    print("WiFi 已连接")
    print("IP 地址:", wlan.ifconfig()[0])


# 主程序
if __name__ == "__main__":
    # WiFi 配置
    WIFI_SSID = "Xiaomi 15 Pro"  # 替换为你的 WiFi 名称
    WIFI_PASSWORD = "xiaomi15pro"  # 替换为你的 WiFi 密码
    connect_wifi(WIFI_SSID, WIFI_PASSWORD)

    # 设置录音参数
    total_samples = 16000 * 5  # 录制5秒的音频（假设采样率为16000 Hz）

    # 开始录音
    print("开始录音...")
    record_audio(total_samples)
    print("录音完成，文件已保存为audio_data.pcm")

    # 发送文件到PC端
    print("正在发送文件到PC端...")
    send_file_to_pc(host="192.168.127.62", port=12345)  # 替换为PC的实际IP地址
    print("文件发送完成")
