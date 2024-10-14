import cv2
import numpy as np
#读取图片
img=cv2.imread('C:/Users/admin/Desktop/sucai1/lena.jpg')
x=img.shape[0]
y=img.shape[1]
#卷积核
kernel=np.ones((5,5),np.float32)/25
#2D卷积
dst=cv2.filter2D(img,-1,kernel)
#图片显示
cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img',int(y),int(x))
cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()