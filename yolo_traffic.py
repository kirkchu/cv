import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO('model/yolo11n.pt')
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture('https://tcnvr3.taichung.gov.tw/39ad6688')
# cap = cv2.VideoCapture('https://cctvs.freeway.gov.tw/live-view/mjpg/video.cgi?camera=1030')
while True:
    ret, image = cap.read()
    if not ret:
        continue
    #results = model(image, device='mps')
    results = model(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_image = box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    cv2.imshow('win', annotated_image)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()