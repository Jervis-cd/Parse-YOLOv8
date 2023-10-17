from ultralytics import YOLO

model=YOLO("/media/ps/work/Parse-YOLOv8/yolov8n.pt",task="detect")

model.export(format="onnx",    # (str) format to export to, choices at https://docs.ultralytics.com/modes/export/#export-formats
             keras=False,      # (bool) use Kera=s
             optimize=False,   # (bool) TorchScript: optimize for mobile
             int8=False,       # (bool) CoreML/TF INT8 quantization 
             dynamic=False,    # (bool) ONNX/TF/TensorRT: dynamic axes
             simplify=False,   # (bool) ONNX: simplify model
             opset=12,         # (int, optional) ONNX: opset version
             workspace=4,      # (int) TensorRT: workspace size (GB)
             nms=False         # (bool) CoreML: add NMS)
             )
