import cv2
import numpy as np

img = cv2.imread('./dcz.jpg')
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lower = np.array([78,43,46])
upper = np.array([124,255,255])
mask = cv2.inRange(img1,lower,upper)
cv2.imshow('mask',mask)
blue = cv2.bitwise_and(img,img,mask=mask)
cv2.imshow('blue',blue)


cv2.waitKey(0)
cv2.destroyAllWindows()
