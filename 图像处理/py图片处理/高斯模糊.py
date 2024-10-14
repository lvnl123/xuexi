import cv2
import numpy as np
#读取图片
img=cv2.imread('C:/Users/admin/Desktop/sucai1/lena.jpg')
x=img.shape[0]
y=img.shape[1]
#高斯模糊,让函数自己计算相关标准差
dst=cv2.GaussianBlur(img,(11,11),0)
#图片显示
cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img',int(y),int(x))
cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
