import os
from ultralytics import YOLO

def train_yolo(model_path, data_path, save_path, epochs, img_size, batch_size, device):
    model = YOLO(model_path)
    model.train(
        project=save_path,
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        plots=True)

if __name__ == '__main__':
    # Configuration
    model_path = 'yolo11s.pt'
    data_path = 'yolo_ds/data.yaml'
    save_path = 'runs'
    epochs = 100
    img_size = 640
    batch_size = 0.8  # -1 for auto, 0.8 for 80% GPU memory
    device = "0"  # "0" for GPU, "cpu" for CPU

    train_yolo(model_path, data_path, save_path, epochs, img_size, batch_size, device)