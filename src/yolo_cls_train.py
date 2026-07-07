from ultralytics import YOLO
model = YOLO('src/../model/yolo26n-cls.pt')
save_dir = 'src/../model/runs/classify'
model.train(
    data='src/clsdataset',
    epochs=50,
    imgsz=224,
    batch=32,
    project=save_dir,
    device='mps'
)

