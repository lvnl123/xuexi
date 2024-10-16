import cv2
import numpy as np
#读取图片
img=cv2.imread('1.jpg')
x=img.shape[0]
y=img.shape[1]
#化为灰度图
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#卷积核
kernel=np.ones((5,5),np.float32)/25
#2D卷积
dst=cv2.filter2D(grey,-1,kernel)
#阀值操作
ret,thresh=cv2.threshold(dst,127,255,cv2.THRESH_BINARY)
#图像显示
if ret:
    cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img',int(y),int(x))
    cv2.imshow('img',thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('二值化失败！')