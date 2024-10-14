import cv2
import numpy as np

img1 = np.zeros((256,256,3),dtype=np.uint8)
img2 = np.zeros((256,256,3),dtype=np.uint8)

cv2.rectangle(img1,(50,50),(200,200),(255,255,255),-1)
cv2.circle(img2,(128,128),100,(255,255,255),-1)
result1 = cv2.bitwise_and(img1,img2)
result2 = cv2.bitwise_not(img1)
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.imshow('result',result1)
cv2.imshow('result1',result2)
cv2.waitKey(0)
cv2.destroyAllWindows()

