import cv2

img=cv2.imread('sucai1/lena.jpg')
if img is None:
    print('no img')
else:
    cv2.imshow('lena',img)
    key=cv2.waitKey(0)
    if key&0xff==ord('q'):
        exit(0)
    elif key&0xff==ord('x'):
        cv2.imwrite('lena2.jpg',img)
        cv2.destroyAllWindows()
print('000')


cv2.namedWindow('456664',cv2.WINDOW_NORMAL)
cv2.resizeWindow('456664',800,600)
cv2.imshow('456664',img)
cv2.waitKey(0)
print(img.shape)

img = cv2.imread('sucai1/dcz.jpg')
cv2.namedWindow('456664',cv2.WINDOW_NORMAL)


h,w,c = img.shape
cv2.resizeWindow('456664',w//2,h//2)
cv2.imshow('456664',img)
cv2.waitKey(0)
print(img.dtype)
print(img.size)





