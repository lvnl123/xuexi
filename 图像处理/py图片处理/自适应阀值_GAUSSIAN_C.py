import cv2
import numpy as np
img=cv2.imread('1.jpg')
#获取图像属性
x=img.shape[0]
y=img.shape[1]
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#阀值操作
thresh=cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2 )
#图像显示
cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img',int(y/2),int(x/2))
cv2.imshow('img',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()