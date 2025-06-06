from ultralytics import YOLO
import cv2

# url = 'https://cctv-ss04.thb.gov.tw/T14A-d61a0c91'
url = 'data/vtest.avi'
model = YOLO('model/yolo11m-pose.pt')
cap = cv2.VideoCapture(url)
while True:
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture(url)
        continue
    # 影像 resize 1.5 倍
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    results = model(frame, verbose=False)[0]
    annotated_frame = results.plot()
    cv2.imshow('YOLOv8 Pose Detection', annotated_frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
