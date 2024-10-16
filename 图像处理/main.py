import cv2
import numpy as np

img = np.zeros((600, 800, 3), dtype=np.uint8)
def nothing(x):
    pass

windowName = 'RBG调色'
cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('R', 'img', 0, 255, nothing)
cv2.createTrackbar('G', 'img', 0, 255, nothing)
cv2.createTrackbar('B', 'img', 0, 255, nothing)

while(1):
    cv2.imshow('img', img)
    k = cv2.waitKey(1) & 0xFF
    if k ==27:
        break
    r = cv2.getTrackbarPos('R', 'img')
    g = cv2.getTrackbarPos('G', 'img')
    b = cv2.getTrackbarPos('B', 'img')

    img[:] = [b, g, r]

    cv2.imshow('img', img)


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()