import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


# supervision 0.29.1 的 get_polygon_center 使用 np.cross 處理 2D 多邊形，
# 在 numpy 2.x 會因維度檢查而拋錯。這裡在建立 PolygonZoneAnnotator 前動態修復。
def _fixed_get_polygon_center(polygon: np.ndarray):
    if len(polygon) == 0:
        raise ValueError("Polygon must have at least one vertex.")
    shift_polygon = np.roll(polygon, -1, axis=0)
    # 2D 叉積的 z 分量，即 shoelace 公式的有向面積
    signed_areas = (
        polygon[:, 0] * shift_polygon[:, 1]
        - polygon[:, 1] * shift_polygon[:, 0]
    ) / 2.0
    if signed_areas.sum() == 0:
        center = np.mean(polygon, axis=0).round()
        return sv.Point(x=center[0], y=center[1])
    centroids = (polygon + shift_polygon) / 3.0
    center = np.average(centroids, axis=0, weights=signed_areas).round()
    return sv.Point(x=center[0], y=center[1])


# polygon_zone 模組在匯入時已綁定原函式，必須同時置換模組內的參考
import supervision.geometry.utils as _geometry_utils
import supervision.detection.tools.polygon_zone as _polygon_zone

_geometry_utils.get_polygon_center = _fixed_get_polygon_center
_polygon_zone.get_polygon_center = _fixed_get_polygon_center

model = YOLO('src/../model/yolo26n-seg.pt')
tracker = sv.ByteTrack()
mask_annotator = sv.MaskAnnotator()
trace_annotator = sv.TraceAnnotator()
label_annotator = sv.LabelAnnotator(
    text_scale=0.4, text_padding=3, text_position=sv.Position.CENTER_OF_MASS)

zone = np.array([(250, 200), (50, 530), (700, 530), (500, 200)])
polygon_zone = sv.PolygonZone(polygon=zone)
polygon_annotator = sv.PolygonZoneAnnotator(
    zone=polygon_zone, color=sv.Color.RED)

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