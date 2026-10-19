from ultralytics import YOLO

# Load your trained model
model = YOLO(r"runs/detect/train-8/weights/best.pt")

# Run inference on a video
results = model.predict(
    source="ppe.mp4",
    conf=0.35,
    show=True,
    save=True
)
