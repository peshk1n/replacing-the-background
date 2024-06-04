import cv2
import numpy as np

min_contour_area = 1000

image = cv2.imread('images/img4.jpg')
image = cv2.resize(image, (500, 500))
cv2.imshow('Image', image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.medianBlur(gray, 3)

canny = cv2.Canny(gray, 25, 175)
cv2.imshow('Canny', canny)

edges = cv2.dilate(canny, None)
edges = cv2.erode(edges, None)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

mask = np.zeros(image.shape[:2], np.uint8)

for contour in contours:
    if cv2.contourArea(contour) > min_contour_area:
        mask = cv2.fillConvexPoly(mask, contour, (255))


kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.medianBlur(mask, 9)

cv2.imshow('Mask', mask)

result = cv2.bitwise_or(image, image, mask=mask)
cv2.imshow('Result', result)
cv2.imwrite('images/result.jpg', result)

cv2.waitKey(0)