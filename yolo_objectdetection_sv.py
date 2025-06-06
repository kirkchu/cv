import cv2
import supervision as sv
from ultralytics import YOLO

image = cv2.imread('data/bus.jpg')
model = YOLO("model/yolov8n.pt")
results = model(image)[0]

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

detections = sv.Detections.from_ultralytics(results)
annotated_image = box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

cv2.imshow('win', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
