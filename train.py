from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(
    data="/home/ubuntu/kitti-yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="kitti_yolo11s",
    project="/home/ubuntu/kitti-yolo/runs"
)
