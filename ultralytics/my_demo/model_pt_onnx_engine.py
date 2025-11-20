from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("/home/jetson/code/ultralytics/ultralytics/data/export/yolo11n.pt")
# model = YOLO("/home/jetson/code/ultralytics/ultralytics/data/export/yolo11n-seg.pt")
# model = YOLO("/home/jetson/code/ultralytics/ultralytics/data/export/yolo11n-pose.pt")
# model = YOLO("/home/jetson/code/ultralytics/ultralytics/data/export/yolo11n-cls.pt")
# model = YOLO("/home/jetson/code/ultralytics/ultralytics/data/export/yolo11n-obb.pt")

# Export the model to TensorRT
# model.export(format="engine", dynamic=True, imgsz=32)
# model.export(format="engine", half=True, dynamic=True, imgsz=32)
# model.export(format="engine", int8=True, dynamic=True, imgsz=32)
# model.export(format="engine", int8=True, data="coco128.yaml", fraction=1.0, dynamic=True, imgsz=32)
# model.export(format="engine", int8=True, data="coco128.yaml", fraction=1.0, dynamic=False, imgsz=32)
model.export(format="engine", int8=True, device="dla:0", data="coco128.yaml", fraction=1.0, dynamic=False, imgsz=32)
