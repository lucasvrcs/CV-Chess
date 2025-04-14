# CV-Chess
This repository contains the implementation of a computer vision project focused on building a complete system to analyze and detect chess piece positions.


## Objective
1. Detect board corners / squares
2. Detect and classify chess pieces on a real-world chessboard image.


## Pipeline Overview
### 1. Chess Piece Detection and Classification
- Uses OpenCV-based image processing techniques to detect the board and individual squares.
- Identifies pieces using template matching and classical methods.

### 2. Board Reconstruction
- Converts the layout of the board into 2d.
- Sorts pieces into their corresponding squares.
- Handles board orientation and alignment.


## Dataset
- Input images of real-world chessboards.
- 50 images.
- Different illuminations.
- Different view angles.


## Project Structure

```
.
├── src/
│   ├── board_detection.py        # Detects the board and squares using OpenCV
│   ├── piece_identifier.py       # Identifies pieces using templates
│   ├── fen_generator.py          # Converts board state into FEN notation
│   └── move_predictor.py         # Predicts best move from a given FEN
│
├── data/
│   ├── images/                   # Sample input images
│   └── templates/                # Template images for each piece
│
├── notebooks/
│   ├── detection_pipeline.ipynb  # Interactive notebook for piece detection
│   └── move_prediction.ipynb     # Notebook for evaluating move prediction
│
├── models/                       # Trained policy models
├── inference_pipeline.py         # Complete pipeline: image → FEN → move prediction
├── requirements.txt              # Environment dependencies
└── README.md                     # This file
```


## Evaluation
- Accuracy of board state reconstruction (visually inspected).
- Accuracy of piece positions (visually inspected).


## How to Use
### 1. Install requirements:
```bash
pip install -r requirements.txt
```

### 2. Run detection on a new image:
```bash
python inference_pipeline.py --image path_to_chess_image.jpg
```

### 3. Output will include:
- Detected board layout
- Detected occupancy grid
- Detected piece positions
- Json format


## Notes
- Board detection uses line detection and contour approximation.
- Occupancy grid is used for piece recognition.


## Authors
Project developed for the Computer Vision and AI course at FEUP/FCUP 2024/2025.

- Lucas Santiago
- Daniel Dias
- Rafael Conceição
- Nuno Moreira


## Future Work
- Improve robustness to lighting and occlusion.
- Improve piece occupancy detection.
