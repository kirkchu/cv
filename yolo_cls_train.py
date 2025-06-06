from ultralytics import YOLO
model = YOLO('model/yolo11n-cls.pt')
save_dir = 'model/runs/classify'
model.train(
    data='dataset',
    epochs=50,
    imgsz=224,
    batch=32,
    project=save_dir,
    device='mps'
)

