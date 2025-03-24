import cv2
import os
import time

# 获取合法学号输入
while True:
    user_input = input("请输入学号（1-50）：")
    try:
        userid = int(user_input)
        if 1 <= userid <= 50:
            break
        else:
            print("学号不在有效范围内，请重新输入。")
    except ValueError:
        print("请输入有效的数字。")

# 创建存储目录
os.makedirs('databet', exist_ok=True)

# 初始化人脸检测器
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 设置分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

sample_num = 0
MAX_SAMPLES = 50

try:
    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频流")
            break

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            maxSize=(300, 300)  # 最大人脸尺寸（可选）
        )

        # 处理每个检测到的人脸
        for (x, y, w, h) in faces:
            # 截取人脸区域（保留彩色信息）
            face_roi = frame[y:y + h, x:x + w]

            # 将人脸区域转换为灰度图
            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # 生成文件名
            timestamp = int(time.time())
            filename = os.path.join("databet", f"user_{userid}_{sample_num}_{timestamp}.png")

            # 保存灰度图像（不包含矩形框）
            try:
                cv2.imwrite(filename, face_roi_gray)
                print(f"已保存灰度图像：{filename}")
            except Exception as e:
                print(f"保存文件时出错：{e}")

            sample_num += 1

            # 绘制矩形框标记人脸（仅用于显示，不影响保存的图像）
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示实时画面（包含矩形框）
        cv2.imshow('Face Capture', frame)

        # 按q键退出
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # 达到最大样本数后退出
        if sample_num >= MAX_SAMPLES:
            print(f"已达到最大样本数 ({MAX_SAMPLES})，正在退出...")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
