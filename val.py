from ultralytics import YOLO

model=YOLO("/media/ps/work/Parse-YOLOv8/yolov8n.pt")

model.val(save_json=False,      # (bool) save results to JSON file
          save_hybrid=False,    # (bool) save hybrid version of labels (labels + additional predictions)
          conf=0.25,            # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
          iou=0.7,              # (float) intersection over union (IoU) threshold for NMS
          max_det=300,          # (int) maximum number of detections per image
          half=False,           # (bool) use half precision (FP16)
          dnn=False,            # (bool) use OpenCV DNN for ONNX inference
          plots=True            # (bool) save plots during train/val
          )