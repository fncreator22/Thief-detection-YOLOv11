
from ultralytics import YOLO  # Import YOLOv8 library

# Step 1: Load a pre-trained YOLOv8 model
# 'yolov8n.pt' = nano version (fastest, smallest, uses less CPU/GPU)
# You can change to 'yolov8s.pt', 'yolov8m.pt', etc., for larger models if you have enough resources
model = YOLO('yolov8n.pt')

# Step 2: Train the model
# data='data.yaml'   → path to your dataset configuration file
# epochs=50          → number of full passes through the dataset
# imgsz=640          → image size (resolution for training)
# batch=8            → number of images processed at once (lower for CPU)
# device='cpu'       → force training on CPU if you don’t have GPU
model.train(
    data='data.yaml',
    epochs=2,
    imgsz=320,
    batch=6,
    device='cpu'
)

# Step 3: Save model results
# Training will create a 'runs/detect/train' folder containing:
# - weights/best.pt   → best-performing model
# - weights/last.pt   → last saved model after final epoch
# - results.png       → training performance graphs
# - metrics.json      → performance numbers
