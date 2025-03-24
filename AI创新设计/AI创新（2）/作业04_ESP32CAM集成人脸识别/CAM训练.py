import os
import cv2
import numpy as np

# 1. 初始化 LBPHFaceRecognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 2. 加载数据集
def load_dataset(dataset_path):
    faces = []
    ids = []

    for image_name in os.listdir(dataset_path):
        if image_name.endswith(".png"):
            # 提取学号（ID）
            parts = image_name.split("_")
            label = int(parts[1])  # 学号作为标签

            # 读取图像
            image_path = os.path.join(dataset_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                faces.append(image)
                ids.append(label)

    # 将 faces 和 ids 转换为 NumPy 数组
    faces = np.array(faces, dtype=object)  # 图像列表
    ids = np.array(ids)  # 标签列表

    return faces, ids

dataset_path = "facedata"  # 数据集路径
faces, ids = load_dataset(dataset_path)

# 检查数据是否为空
if len(faces) == 0 or len(ids) == 0:
    print("数据集为空，请检查数据路径或采集数据！")
    exit()

# 3. 训练模型
print("开始训练模型...")
recognizer.train(faces, ids)

# 4. 保存模型
model_path = "face_recognizer_model.yml"
recognizer.save(model_path)
print(f"模型已保存到 {model_path}")
