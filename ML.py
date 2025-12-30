from ultralytics import YOLO


model = YOLO("yolov8n.pt")
model.train(data="dudataset/data.yaml", epochs=150, imgsz=640, patience=20, batch=2, name="yolometalVh")