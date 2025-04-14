# Chess Piece Detection

This repository contains the implementation of a computer vision project focused on chess board analysis and piece presence detection. The project uses classical computer vision techniques without relying on deep learning models.

## Project Objective

The main goals of this project are:
1. Detect the chess board and its corners in images taken from various angles
2. Identify the presence of chess pieces on the board
3. Generate a standardized output showing the board state as an 8×8 matrix and piece locations

## Implementation Overview

### Chess Detection Pipeline

The system follows a sequential processing pipeline:

1. **Board Detection**: Identifies the chess board using edge detection, line extraction, and geometric filtering
2. **Square Extraction**: Divides the board into 64 equal squares with proper chess notation (a1-h8)
3. **Corner Identification**: Locates the four corners of the board using convex hull and angle-based methods
4. **Perspective Transformation**: Warps the image to obtain a top-down view of the board
5. **Piece Detection**: Analyzes each square to determine piece presence using image features
6. **Output Generation**: Creates a standardized JSON output with piece counts and positions

## Usage

### Input Format

The system takes a JSON file as input specifying image paths:

```json
{
  "image_files": [
    "images/G000_IMG062.jpg",
    "images/G000_IMG087.jpg",
    "images/G033_IMG043.jpg"
  ]
}
```

### Running the Detection

```bash
python ChessDetection.py
```

This processes all images listed in `input.json` and generates:
1. Visualization outputs in the `processed_boards` folder
2. Detection results in `output.json`

### Output Format

The program generates an `output.json` file with the following structure:

```json
[
  {
    "image": "images/G000_IMG062.jpg",
    "num_pieces": 32,
    "board": [
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1]
    ],
    "detected_pieces": [
      {
        "xmin": 100,
        "ymin": 200,
        "xmax": 150,
        "ymax": 250
      },
      // More pieces...
    ]
  },
  // More images...
]
```

Where:
- `image`: Path to the processed image
- `num_pieces`: Total number of pieces detected
- `board`: 8×8 matrix where 1 indicates piece presence, 0 indicates empty square
- `detected_pieces`: List of bounding boxes for each detected piece

## Key Technical Features

- **Hand-tuned Canny Edge Detection**: Uses unconventional thresholds (250/180) for optimal chess grid extraction
- **Optimized HoughLinesP**: Parameters (threshold=730, minLineLength=225, maxLineGap=110) specifically calibrated for chess boards
- **Geometric Square Filtering**: Uses precise constraints on aspect ratio, diagonal ratio, and size to identify valid chess squares
- **Feature-based Piece Detection**: Combines brightness variation and edge density to reliably detect pieces regardless of color

## Project Structure

```
.
├── ChessDetection.py      # Main detection script
├── input.json             # Input file listing images to process
├── images/                # Directory containing chess board images
├── processed_boards/      # Output directory for visualizations
└── output.json            # Detection results in JSON format
```

## Requirements
- OpenCV
- NumPy
- Matplotlib

## Authors

Project developed for the Computer Vision course at FEUP 2024/2025.
- Lucas Santiago
- Daniel Dias
- Rafael Conceição
- Nuno Moreira
