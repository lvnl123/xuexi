import cv2
import socket
import numpy as np
import time

# ESP32-CAM 配置参数
ESP32_IP = "192.168.42.3"  # 替换为你的 ESP32-CAM 实际 IP 地址
ESP32_PORT = 8080         # ESP32-CAM 的 TCP 端口号
TIMEOUT = 5               # 超时时间（秒）

# 加载训练好的模型
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer_model.yml")

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 定义目标用户 ID 和置信度阈值
TARGET_USER_ID = 26  # 目标用户学号
CONFIDENCE_THRESHOLD = 70  # 置信度阈值（越低表示匹配越好）

# 从 ESP32-CAM 获取单帧图像（通过 TCP）
def get_esp32_image():
    """通过 TCP 从 ESP32-CAM 获取单帧图像"""
    try:
        # 创建 TCP 客户端套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TIMEOUT)
        sock.connect((ESP32_IP, ESP32_PORT))
        print("已连接到 ESP32-CAM TCP 服务器")

        # 接收图像大小
        img_size_data = sock.recv(4)
        if len(img_size_data) != 4:
            print("接收图像大小失败")
            sock.close()
            return None

        img_size = int.from_bytes(img_size_data, byteorder='little')
        print(f"接收图像大小：{img_size} 字节")

        # 接收图像数据
        img_data = b""
        while len(img_data) < img_size:
            packet = sock.recv(img_size - len(img_data))
            if not packet:
                print("接收图像数据失败")
                sock.close()
                return None
            img_data += packet

        # 关闭套接字
        sock.close()

        # 解码图像
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            print("图像解码失败")
        return frame

    except socket.error as e:
        print(f"网络错误: {e}")
        return None

# 打开电脑摄像头
cap = cv2.VideoCapture(0)

# 初始化变量
use_esp32_cam = True  # 默认使用 ESP32-CAM

# 打印操作提示信息
print("开始采集人脸图像，按 Q 键停止，按 C 键切换摄像头源...")
print(f"当前摄像头源：{'ESP32-CAM' if use_esp32_cam else '电脑摄像头'}")

while True:
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

    for (x, y, w, h) in faces:
        # 提取人脸区域
        face_roi = gray[y:y+h, x:x+w]

        # 预测
        label, confidence = recognizer.predict(face_roi)

        # 打印当前识别结果和置信度
        print(f"识别结果：User {label}, 置信度：{confidence:.2f}")

        # 判断是否为目标用户
        if label == TARGET_USER_ID and confidence < CONFIDENCE_THRESHOLD:
            text = f"User {label} ({confidence:.2f})"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            print(f"成功匹配目标用户: User {label}")
        else:
            text = f"Unknown ({confidence:.2f})"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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
