import cv2
import numpy as np
#读取图片
img=cv2.imread('1.jpg')
x=img.shape[0]
y=img.shape[1]
#平均卷积
dst=cv2.blur(img,(11,11))
#图片显示
cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img',int(y),int(x))
cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
