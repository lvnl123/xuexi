import cv2
import numpy as np
img=cv2.imread('1.jpg')
x=img.shape[0]
y=img.shape[1]
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(grey,127,255,cv2.THRESH_BINARY)
cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img',y,x)
cv2.imshow('img',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()