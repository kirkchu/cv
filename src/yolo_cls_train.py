import os
from ultralytics import YOLO

save_dir = os.path.abspath('model/runs/classify')
last_weights = os.path.join(save_dir, 'train', 'weights', 'last.pt')
initial_weights = last_weights if os.path.exists(last_weights) else 'model/yolo26n-cls.pt'

model = YOLO(initial_weights)
model.train(
    data='src/clsdataset',
    epochs=100,
    imgsz=224,
    batch=32,
    project=save_dir,
    device='mps',
    resume=True,
)

