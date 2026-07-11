from ultralytics import YOLO

# Load a model
model = YOLO("../model/yolo26n.pt")

# Train using NDJSON dataset
results = model.train(data="src/gshock.ndjson", epochs=100, imgsz=640, device='mps')