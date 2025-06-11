import os
from ultralytics import RTDETR

def train_rtdetr(model_path, data_path, save_path, epochs, img_size, batch_size, device):
    model = RTDETR(model_path)
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
    model_path = 'rtdetr-l.pt'
    data_path = 'yolo_ds/data.yaml'
    save_path = 'runs/rt'
    epochs = 100
    img_size = 640
    batch_size = 0.8
    device = "0"

    train_rtdetr(model_path, data_path, save_path, epochs, img_size, batch_size, device)