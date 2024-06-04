import cv2
import numpy as np

img = cv2.imread('images/img3.jpg')
img = cv2.resize(img, [500, 500])
cv2.imshow('Image', img)

mask = np.zeros(img.shape[:2], np.uint8)

rect = (40, 40, img.shape[1]-40, img.shape[0]-40)
rectangle = cv2.rectangle(img.copy(), rect[:2], rect[2:], 255, 3)
cv2.imshow('Rectangle', rectangle)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 15, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

white_bg = np.ones_like(img, dtype=np.uint8) * 255
mask_inv = cv2.bitwise_not(mask2 * 255)

img_fg = cv2.bitwise_and(img, img, mask=mask2)
bg_fg = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)

result = cv2.add(img_fg, bg_fg)
cv2.imshow('Result 1', result)
cv2.imwrite('images/result.jpg', result)

newmask = cv2.imread('images/img3_mask.png', cv2.IMREAD_GRAYSCALE)
newmask = cv2.resize(newmask, [500, 500])
mask[newmask == 0] = 0
mask[newmask == 255] = 1

final_mask, _, _ = cv2.grabCut(result, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

mask2 = np.where((final_mask == 2) | (final_mask == 0), 0, 1).astype('uint8')

white_bg = np.ones_like(img, dtype=np.uint8) * 255
mask_inv = cv2.bitwise_not(mask2 * 255)

img_fg = cv2.bitwise_and(img, img, mask=mask2)
bg_fg = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)

final_result = cv2.add(img_fg, bg_fg)
cv2.imshow('Result 2', final_result)
cv2.imwrite('images/result.jpg', final_result)
cv2.waitKey(0)
