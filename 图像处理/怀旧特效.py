import cv2
import numpy as np
img = cv2.imread('sucai4/dcz.jpg')
h,w,c = img.shape
for i in range(h):
    for j in range(w):
        b = img[i,j][0]
        g = img[i,j][1]
        r = img[i,j][2]
        R = 0.393*r + 0.769*g + 0.189*b
        G = 0.349*r + 0.686*g + 0.168*b
        B = 0.272*r + 0.534*g + 0.131*b
        img[i,j][0] = max(0,min(B,255))
        img[i,j][1] = max(0,min(G,255))
        img[i,j][2] = max(0,min(R,255))

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()