import os
from ultralytics import YOLO

model_path = os.path.abspath('model/yolo26n-cls.pt')
dataset_path = os.path.abspath('clsdataset')
save_dir = os.path.abspath('model/runs/classify')

model = YOLO(model_path)
model.train(
    data=dataset_path,
    project=save_dir,
    epochs=50,
    imgsz=224,
    batch=32,
    device='mps'
)

