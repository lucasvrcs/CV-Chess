# Task 3: Chess Piece Detection

This project implements chess piece detection using YOLO and RT-DETR object detection models. It detects chess pieces on board images and creates a digital twin representation showing piece positions in standard chess notation (a1-h8).

## Files

- `yolo_train.py` - Train YOLO model
- `rt_train.py` - Train RT-DETR model
- `yolo_script.py` - YOLO detection
- `rt_script.py` - RT-DETR detection
- `task_3_yolo.ipynb` - Complete YOLO pipeline
- `task_3_benchmark.ipynb` - Performance analysis of both models

## Usage

### Train Models
```bash
python yolo_train.py
python rt_train.py
```

### Run Detection
```python
# YOLO detection
from yolo_script import detect_chess_pieces
fig, result = detect_chess_pieces('image.jpg')

# RT-DETR detection
from rt_script import detect_chess_pieces
fig, result = detect_chess_pieces('image.jpg')
```

Or run the scripts directly:
```bash
python yolo_script.py
python rt_script.py
```

### Compare Models
```bash
# Run the task_3_benchmark.ipynb file
```

## Dataset
Uses YOLO format in `yolo_ds/` folder with 12 chess piece classes.
