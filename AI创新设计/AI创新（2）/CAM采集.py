import cv2
import os
import time
import requests
import numpy as np

# ESP32-CAM配置参数
ESP32_IP = "192.168.42.3"  # 替换为你的ESP32-CAM实际IP地址
CAPTURE_URL = f"http://{ESP32_IP}/capture"
TIMEOUT = 5  # 请求超时时间（秒）

# 创建存储目录
os.makedirs('face_data', exist_ok=True)

# 初始化人脸检测器
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def get_esp32_image():
    """从ESP32-CAM获取单帧图像"""
    try:
        response = requests.get(CAPTURE_URL, timeout=TIMEOUT)
        if response.status_code == 200:
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            print(f"HTTP错误码: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"网络请求失败: {e}")
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

    print("开始采集人脸图像，按Q键停止...")

    while sample_num < MAX_SAMPLES:
        # 获取图像
        frame = get_esp32_image()
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
            filename = f"face_data/user_{userid}_{sample_num}_{timestamp}.png"

            cv2.imwrite(filename, face_img)
            print(f"已保存: {filename}（样本 {sample_num + 1}/{MAX_SAMPLES}）")
            sample_num += 1

            # 在显示画面绘制矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示实时画面
        cv2.imshow('ESP32-CAM Face Capture', frame)

        # 按Q键提前退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("采集完成！")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()