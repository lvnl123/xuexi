import numpy as np
import cv2

def nothing(x):
    pass
img = cv2.imread('./sucai4/bai.jpeg',0)
windowName = "dcz_Threshoulding"
cv2.namedWindow(windowName,cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('Type',windowName,0,4,nothing)
cv2.createTrackbar('Value',windowName,0,255,nothing)
while(1):
    if cv2.waitKey(1) & 0xFF == 27:
        break
    Type = cv2.getTrackbarPos('Type',windowName)
    Value = cv2.getTrackbarPos('Value',windowName)
    ret, dst = cv2.threshold(img,Value,255,Type)
    cv2.imshow(windowName, dst)
cv2.destroyAllWindows()