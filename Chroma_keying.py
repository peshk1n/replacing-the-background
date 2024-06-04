import cv2
import numpy as np

def replace_bg(img, bg):
    bg = cv2.resize(bg, (img.shape[1], img.shape[0]))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 60, 40])
    upper_green = np.array([70, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 7)

    mask_inv = cv2.bitwise_not(mask)

    foreground = cv2.bitwise_and(img, img, mask=mask_inv)
    background = cv2.bitwise_and(bg, bg, mask=mask)

    result = cv2.bitwise_or(foreground, background)

    return result


cap = cv2.VideoCapture('videos/video1.mp4')
bg = cv2.imread('images/bg1.jpg')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))
out = cv2.VideoWriter('videos/video2.mp4',
                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (800, 450))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Ошибка: Не удалось прочитать кадр.")
        break

    frame = cv2.resize(frame, (800, 450))
    replaced = replace_bg(frame, bg)

    cv2.imshow('Origin', frame)
    cv2.imshow('Result', replaced)

    out.write(frame.astype('uint8'))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()