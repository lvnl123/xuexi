import cv2
import os
import time
import socket
import numpy as np

# ESP32-CAM 配置参数
ESP32_IP = "192.168.42.3"  # 替换为你的 ESP32-CAM 实际 IP 地址
ESP32_PORT = 8080         # ESP32-CAM 的 TCP 端口号
TIMEOUT = 5               # 超时时间（秒）

# 创建存储目录
os.makedirs('facedata', exist_ok=True)

# 初始化人脸检测器
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def get_esp32_image():
    """通过 TCP 从 ESP32-CAM 获取单帧图像"""
    try:
        # 创建 TCP 客户端套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TIMEOUT)
        sock.connect((ESP32_IP, ESP32_PORT))

        # 接收图像大小
        img_size_data = sock.recv(4)
        if len(img_size_data) != 4:
            print("接收图像大小失败")
            sock.close()
            return None

        img_size = int.from_bytes(img_size_data, byteorder='little')

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
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    except socket.error as e:
        print(f"网络错误: {e}")
        return None


def main():
    # 获取学号输入
    while True:
        try:
            userid = int(input("请输入学号（1-50）："))
            if 1 <= userid <= 50:
                break
            print("学号超出范围")
        except ValueError:
            print("请输入有效数字")

    sample_num = 0
    MAX_SAMPLES = 50

    # 初始化电脑摄像头
    cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
    if not cap.isOpened():
        print("无法打开电脑摄像头！")
        return

    print("开始采集人脸图像，按 Q 键停止，按 C 键切换摄像头源...")

    use_esp32_cam = True  # 默认使用 ESP32-CAM
    while sample_num < MAX_SAMPLES:
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

        if frame is None:
            time.sleep(0.1)
            continue

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            maxSize=(300, 300)
        )

        # 处理检测到的人脸
        for (x, y, w, h) in faces:
            # 截取并保存人脸区域
            face_img = gray[y:y + h, x:x + w]
            timestamp = int(time.time() * 1000)
            filename = f"facedata/user_{userid}_{sample_num}_{timestamp}.png"

            cv2.imwrite(filename, face_img)
            print(f"已保存: {filename}（样本 {sample_num + 1}/{MAX_SAMPLES}）")
            sample_num += 1

            # 在显示画面绘制矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示实时画面
        cv2.putText(frame, f"Source: {source_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Capture', frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按 Q 键退出
            break
        elif key == ord('c'):  # 按 C 键切换摄像头源
            use_esp32_cam = not use_esp32_cam
            print(f"切换到 {'ESP32-CAM' if use_esp32_cam else '电脑摄像头'}")

    print("采集完成！")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
