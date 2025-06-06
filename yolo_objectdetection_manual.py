from ultralytics import YOLO
import cv2

# 初始化 YOLO 模型
model = YOLO('model/yolov8n.pt')  # 使用 YOLOv8 的預訓練模型

# 讀取影像
image_path = 'data/bus.jpg'  # 替換為你的影像路徑
image = cv2.imread(image_path)

# 執行物件偵測
results = model(image)[0]

# 繪製矩形框和物件名稱
for result in results.boxes:
    # 取得邊界框座標
    x1, y1, x2, y2 = map(int, result.xyxy[0])
    # 取得物件名稱
    label = result.cls[0]
    confidence = result.conf[0]
    name = f"{model.names[int(label)]} {confidence:.2f}"
    
    # 繪製矩形框
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 在矩形框上顯示物件名稱
    cv2.putText(image, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 顯示結果影像
cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
