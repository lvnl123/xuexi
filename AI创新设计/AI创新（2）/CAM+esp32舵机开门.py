import cv2
import requests
import numpy as np
import time
import socket
import threading

# ESP32-CAM 配置参数
ESP32_IP = "192.168.110.3"  # 替换为你的 ESP32-CAM 实际 IP 地址
CAPTURE_URL = f"http://{ESP32_IP}/capture"
TIMEOUT = 5  # 请求超时时间（秒）

# 加载训练好的模型
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer_model.yml")

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 定义目标用户 ID 和置信度阈值
TARGET_USER_ID = 26  # 目标用户学号
CONFIDENCE_THRESHOLD = 90  # 置信度阈值（越低表示匹配越好）

# 全局变量
door_open = False  # 记录门是否已打开
cooldown_flag = False  # 冷却标志
last_close_time = None  # 记录门关闭的时间
COOLDOWN_TIME = 5  # 冷却时间为 5 秒
door_lock = threading.Lock()  # 线程锁

# 发送开门指令到 ESP32-CAM 客户端
def send_open_door_command():
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(2)  # 设置超时时间为 2 秒
        client_socket.connect(('192.168.110.241', 5000))  # 替换为 ESP32 的 IP 地址和端口号
        client_socket.sendall(b"OPEN_DOOR")  # 发送开门指令
        response = client_socket.recv(1024).decode('utf-8').strip()
        print(f"服务器响应: {response}")
        client_socket.close()
        return response == "DOOR_ACTION_COMPLETE"
    except Exception as e:
        print(f"发送指令时出错: {e}")
        return False

# 获取单帧图像
def get_esp32_image(max_retries=3):
    """从 ESP32-CAM 获取单帧图像，支持重试"""
    for attempt in range(max_retries):
        try:
            response = requests.get(CAPTURE_URL, timeout=TIMEOUT)
            if response.status_code == 200:
                img_array = np.array(bytearray(response.content), dtype=np.uint8)
                return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            else:
                print(f"HTTP 错误码: {response.status_code}, 尝试 {attempt + 1}/{max_retries}")
        except requests.exceptions.RequestException as e:
            print(f"网络请求失败: {e}, 尝试 {attempt + 1}/{max_retries}")
        time.sleep(0.5)  # 等待一段时间后重试
    return None

# 处理开门动作
def handle_door_action():
    global door_open, cooldown_flag, last_close_time
    with door_lock:
        if door_open or cooldown_flag:
            print("门已打开或处于冷却状态，跳过此次操作")
            return

        print("匹配成功，正在发送开门指令...")
        if send_open_door_command():
            print("门已打开！")
            door_open = True
            time.sleep(3)  # 开门保持 3 秒
            print("延迟 3 秒钟，正在关门...")
            door_open = False  # 关门完成
            cooldown_flag = True  # 启动冷却状态
            last_close_time = time.time()  # 记录门关闭时间
        else:
            print("发送开门指令失败！")

# 打开电脑摄像头
cap = cv2.VideoCapture(0)

# 初始化变量
use_esp32_cam = True  # 默认使用 ESP32-CAM

# 打印操作提示信息
print("开始采集人脸图像，按 Q 键停止，按 C 键切换摄像头源...")
print(f"当前摄像头源：{'ESP32-CAM' if use_esp32_cam else '电脑摄像头'}")

last_frame_time = time.time()

while True:
    current_time = time.time()
    if current_time - last_frame_time < 0.1:  # 每秒最多处理 10 帧
        continue
    last_frame_time = current_time

    # 根据当前选择获取图像
    if use_esp32_cam:
        frame = get_esp32_image()
        source_name = "ESP32-CAM"
    else:
        ret, frame = cap.read()
        source_name = "Computer Camera"
        if not ret:
            print("无法从电脑摄像头读取帧！")
            continue

    # 如果图像获取失败，跳过本次循环
    if frame is None:
        time.sleep(0.1)
        continue

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # 检查冷却状态
    if cooldown_flag and time.time() - last_close_time >= COOLDOWN_TIME:
        print("5秒冷却时间结束，恢复正常检测")
        print('----------------')
        cooldown_flag = False

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(face_roi)

        # 判断是否为目标用户且门已关闭且未处于冷却状态
        if label == TARGET_USER_ID and confidence < CONFIDENCE_THRESHOLD and not door_open and not cooldown_flag:
            text = f"User {label} ({confidence:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            print(f"成功匹配目标用户: User {label}")

            # 启动新线程处理开门动作
            threading.Thread(target=handle_door_action, daemon=True).start()

    # 显示当前摄像头源
    cv2.putText(frame, f"Source: {source_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 显示结果
    cv2.imshow("Face Recognition", frame)

    # 按键处理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 按 Q 键退出
        print("程序已退出。")
        break
    elif key == ord('c'):  # 按 C 键切换摄像头源
        use_esp32_cam = not use_esp32_cam
        current_source = "ESP32-CAM" if use_esp32_cam else "电脑摄像头"
        print(f"切换到 {current_source}")

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("资源已释放")