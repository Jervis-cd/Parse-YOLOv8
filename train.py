from ultralytics import YOLO


model=YOLO("/media/ps/work/Parse-YOLOv8/yolov8n.pt")

model.train()