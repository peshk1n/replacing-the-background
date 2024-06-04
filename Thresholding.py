import cv2
import numpy as np

img = cv2.imread('images/img2.jpg')
background = cv2.imread('images/bg2.jpg')

img = cv2.resize(img, [500, 500])
bg = cv2.resize(background, [500, 500])
cv2.imshow('Image', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def on_trackbar(value):
    threshold = cv2.getTrackbarPos('Threshold', 'Trackbars')

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 7)

    foreground = cv2.bitwise_and(img, img, mask=mask)
    background = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask))

    result = cv2.bitwise_or(foreground, background)

    cv2.imshow('Mask', mask)
    cv2.imwrite('images/mask1.jpg', foreground)
    cv2.imshow('Result', result)
    cv2.imwrite('images/res1.jpg', result)


cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars', 400, 400)
cv2.createTrackbar('Threshold', 'Trackbars', 225, 255, on_trackbar)

on_trackbar(0)

cv2.waitKey(0)
cv2.destroyAllWindows()