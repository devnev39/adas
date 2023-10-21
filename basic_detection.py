import cv2
import time
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture("/Users/dev/Documents/sample.mp4")

while cap.isOpened():
    success, img = cap.read()
    t = time.perf_counter()
    k = cv2.waitKey(5)

    if k == ord('q'):
        break
    result = model(img)
    img = result[0].plot()
    cv2.putText(img, str(round(1/(time.perf_counter() - t),1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Img', img)

cap.release()

cv2.destroyAllWindows()
