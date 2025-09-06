from ultralytics import YOLO

model = YOLO("eaw-yolo11n.yaml")
model.train(data='../URPC2019/data.yaml', epochs=300, batch=16, imgsz=640, device=0)