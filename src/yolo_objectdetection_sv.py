import cv2
import supervision as sv
from ultralytics import YOLO

image = cv2.imread('src/data/bus.jpg')
model = YOLO("model/yolo26n-seg.pt")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

# mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotated_image = mask_annotator.annotate(
    # scene=image, detections=detections)
annotated_image = box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

cv2.imshow('win', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
