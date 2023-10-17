from ultralytics import YOLO

model=YOLO("/media/ps/work/Parse-YOLOv8/yolov8n.pt",task="detect")

model.predict(source="",            # (str, optional) source directory for images or videos
              show=False,           # (bool) show results if possible
              save_txt=False,       # (bool) save results as .txt file
              save_conf=False,      # (bool) save results with confidence scores
              save_crop=False,      # (bool) save cropped images with results
              show_labels=True,     # (bool) show object labels in plots
              show_conf=True,       # (bool) show object confidence scores in plots
              vid_stride=1,         # (int) video frame-rate stride
              stream_buffer=False,  # (bool) buffer all streaming frames (True) or return the most recent frame (False)
              line_width=1,         # (int, optional) line width of the bounding boxes, auto if missing
              visualize=False,      # (bool) visualize model features
              augment=False,        # (bool) apply image augmentation to prediction sources
              agnostic_nms=False,   # (bool) class-agnostic NMS
              classes=[],           # (int | list[int], optional) filter results by class, i.e. classes=0, or classes=[0,2,3]
              retina_masks=False,   # (bool) use high-resolution segmentation masks
              boxes=True            # (bool) Show boxes in segmentation predictions)
              )