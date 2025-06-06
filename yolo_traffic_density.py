import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO('model/yolo11n-seg.pt')
tracker = sv.ByteTrack()
trace_annotator = sv.TraceAnnotator()
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator(
    text_scale=0.4, text_padding=3, text_position=sv.Position.CENTER_OF_MASS)

zone = np.array([(250, 200), (50, 530), (700, 530), (500, 200)])
polygon_zone = sv.PolygonZone(zone)
polygon_annotator = sv.PolygonZoneAnnotator(polygon_zone, color=sv.Color.RED)

url = 'https://tcnvr3.taichung.gov.tw/39ad6688'
cap = cv2.VideoCapture(url)
while True:
    ret, image = cap.read()
    if not ret:
        cap = cv2.VideoCapture(url)
        continue

    results = model(image, device='mps', verbose=False)[0]
    # results = model(image)
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    is_detections_in_zone = polygon_zone.trigger(detections)
    print(polygon_zone.current_count)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_image = mask_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    annotated_image = trace_annotator.annotate(
        scene=annotated_image, detections=detections)
    annotated_image = polygon_annotator.annotate(
        scene=annotated_image)


    cv2.imshow('win', annotated_image)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()