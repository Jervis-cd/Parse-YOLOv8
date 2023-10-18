from ultralytics import YOLO


model=YOLO("/media/ps/work/Parse-YOLOv8/yolov8n.pt",task="detect")

model.train(model="yolov8n.pt",     # (str, optional) path to model file, i.e. yolov8n.pt, yolov8n.yaml
            data="ultralytics/cfg/datasets/coco128.yaml",  # (str, optional) path to data file, i.e. coco128.yaml
            epochs=100,             # (int) number of epochs to train for
            patience=50,            # (int) epochs to wait for no observable improvement for early stopping of training
            batch=16,               # (int) number of images per batch (-1 for AutoBatch)
            imgsz=640,              # (int | list) input images size as int for train and val modes, or list[w,h] for predict and export modes
            save=True,              # (bool) save train checkpoints and predict results
            save_period=-1,         # (int) Save checkpoint every x epochs (disabled if < 1)
            cache=False,            # (bool) True/ram, disk or False. Use cache for data loading
            device=0,               # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
            workers=8,              # (int) number of worker threads for data loading (per RANK if DDP)
            project="",             # (str, optional) project name
            name="",                # (str, optional) experiment name, results saved to 'project/name' directory
            exist_ok=False,         # (bool) whether to overwrite existing experiment
            pretrained=True,        # (bool | str) whether to use a pretrained model (bool) or a model to load weights from (str)
            optimizer="auto",       # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
            verbose=True,           # (bool) whether to print verbose output
            seed=0,                 # (int) random seed for reproducibility
            deterministic=True,     # (bool) whether to enable deterministic mode
            single_cls=False,       # (bool) train multi-class data as single-class
            rect=False,             # (bool) rectangular training if mode='train' or rectangular validation if mode='val'
            cos_lr=False,           # (bool) use cosine learning rate scheduler
            close_mosaic=10,        # (int) disable mosaic augmentation for final epochs (0 to disable)
            resume=False,           # (bool) resume training from last checkpoint
            amp=True,               # (bool) Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check
            fraction=1.0,           # (float) dataset fraction to train on (default is 1.0, all images in train set)
            profile=False,          # (bool) profile ONNX and TensorRT speeds during training for loggers
            freeze=None,            # (int | list, optional) freeze first n layers, or freeze list of layer indices during training
            # Segmentation
            overlap_mask=True,      # (bool) masks should overlap during training (segment train only)
            mask_ratio=4,           # (int) mask downsample ratio (segment train only)
            # Classification
            dropout=0.0             # (float) use dropout regularization (classify train only)
            )