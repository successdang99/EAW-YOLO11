from ultralytics import YOLO
 
model = YOLO('./runs/detect/train10/weights/best.pt')
model.val(data='../URPC2020_1/data.yaml', split='test', batch=16, imgsz=640, device=0)