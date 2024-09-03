

import cv2
import numpy
from Object_detection import ObjectDetection

ob = ObjectDetection()
cap = cv2.VideoCapture("CarDetection.mp4")

count = 0

while True:
    a, frame = cap.read()
    (class_id, scores, boxes) = ob.detect(frame=frame)
    for box in boxes:
        (x, y, w, h) = box
        count = count+1
        cv2.rectangle(frame, (x, y), (w+x, h+y), (30, 255, 156), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

