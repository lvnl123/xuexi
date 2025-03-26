import time
import socket
import network
from machine import Pin, PWM
import _thread

# 配置舵机连接的引脚
servo_pin = Pin(4)  # 假设舵机连接在 GPIO4 引脚上
servo = PWM(servo_pin, freq=50)  # 设置 PWM 频率为 50Hz，适用于多数舵机

def set_servo_angle(angle):
    """
    根据给定的角度设置舵机的位置
    """
    # 舵机脉宽范围：0.5ms ~ 2.5ms（对应 0° ~ 180°）
    duty = int(26 + (angle * 102 / 180))  # 26 对应 0.5ms，128 对应 2.5ms
    servo.duty(duty)
    time.sleep_ms(500)  # 给予时间让舵机完成动作

def open_door():
    """
    模拟开门 - 顺时针转 90°
    """
    print("开门中...")
    set_servo_angle(90)

def close_door():
    """
    模拟关门 - 逆时针转回初始位置
    """
    print("3秒已到，关门中...")
    set_servo_angle(0)

def handle_door_action():
    """
    处理开门和关门操作
    """
    open_door()
    time.sleep(3)  # 开门保持 3 秒
    close_door()

def handle_client_connection(client_socket):
    """
    处理客户端连接并响应开门指令
    """
    try:
        request = client_socket.recv(1024).decode('utf-8').strip()
        if request == "OPEN_DOOR":
            print(f"收到开门指令: {request}")
            # 启动新线程处理开关门操作
            _thread.start_new_thread(handle_door_action, ())
            client_socket.sendall(b"DOOR_ACTION_COMPLETE")
        else:
            print(f"忽略无效指令: {request}")
    except Exception as e:
        print(f"处理客户端时出错: {e}")
    finally:
        client_socket.close()

def start_server():
    """
    启动服务器监听开门指令
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('192.168.110.241', 5000))  # 监听所有网络接口，端口 5000
    server_socket.listen(5)
    print("服务器已启动，等待指令...")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"收到连接: {addr}")
        handle_client_connection(client_socket)

def connect_wifi(ssid, password, timeout=10):
    """
    连接到 WiFi，带超时机制
    """
    sta_if = network.WLAN(network.STA_IF)
    if not sta_if.isconnected():
        print("正在连接 WiFi...")
        sta_if.active(True)
        sta_if.connect(ssid, password)
        start_time = time.time()
        while not sta_if.isconnected():
            if time.time() - start_time > timeout:
                print("WiFi 连接超时")
                return False
            time.sleep(0.5)
    print("WiFi 已连接:", sta_if.ifconfig())
    return True

try:
    connect_wifi("Xiaomi 15 Pro", "xiaomi15pro")
    ip_address = network.WLAN(network.STA_IF).ifconfig()[0]
    print(f"ESP32 的 IP 地址是: {ip_address}")
    start_server()
finally:
    servo.deinit()
    print("资源已释放")