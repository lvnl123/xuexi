import cv2
import numpy as np

img = cv2.imread('./lena1.JPG')
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow('BGR',img)
cv2.imshow('RGB',img1)
cv2.imshow('GRAY',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
