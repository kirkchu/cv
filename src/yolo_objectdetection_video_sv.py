import os
import cv2
import supervision as sv
from pathlib import Path

os.environ['YOLO_VERBOSE'] = 'False'
os.environ['ULTRALYTICS_VERBOSE'] = 'False'

from ultralytics import YOLO

model_path = Path(__file__).resolve().parent.parent / 'model' / 'best.pt'
model = YOLO(str(model_path))

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Cannot open camera')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    cv2.imshow('YOLO Camera Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
