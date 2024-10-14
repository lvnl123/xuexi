import numpy as np
import cv2
img = cv2.imread('./sucai4/letter.png')
blur = cv2.blur(img,(7,7))
cv2.imshow('src',img)
cv2.imshow('blur',blur)
cv2.waitKey(0)
cv2.destroyAllWindows()