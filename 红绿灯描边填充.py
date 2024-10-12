import cv2
import numpy as np

def detect_and_fill_yellow_shapes(image_path):
    # 读取图片
    image = cv2.imread('traffic light.png')

    # 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 黄色阈值范围
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # 定义一个核用于形态学操作
    kernel = np.ones((5, 5), np.uint8)

    # 执行形态学操作以平滑轮廓
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    # 检测轮廓
    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 填充轮廓
    cv2.fillPoly(image, contours_yellow, (0, 255, 255))

    return image

# 使用函数
image_with_fills = detect_and_fill_yellow_shapes('path_to_your_traffic_light_image.jpg')
cv2.imshow('Filled Yellow Shapes', image_with_fills)
cv2.waitKey(0)
cv2.destroyAllWindows()
