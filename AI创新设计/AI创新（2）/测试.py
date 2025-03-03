import cv2

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

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    # 显示结果
    cv2.imshow("Face Recognition", frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()