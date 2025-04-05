import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 3)

def get_grid_cells(img):
    h, w = img.shape[:2]
    cell_h, cell_w = h // 8, w // 8
    cells = []
    for i in range(8):
        for j in range(8):
            x, y = j * cell_w, i * cell_h
            cells.append((x, y, cell_w, cell_h))
    return cells

def detect_pieces(thresh_img, original_img, cells):
    board_matrix = np.zeros((8, 8), dtype=int)
    bounding_boxes = []
    white_pieces = 0
    black_pieces = 0

    for idx, (x, y, w, h) in enumerate(cells):
        cell = thresh_img[y:y+h, x:x+w]
        count = cv2.countNonZero(cell)
        if count > (w * h * 0.1):  # simple presence threshold
            board_matrix[idx // 8, idx % 8] = 1

            # Extract from original for color estimation
            piece_img = original_img[y:y+h, x:x+w]
            mean_color = cv2.mean(cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY))[0]
            piece_color = "white" if mean_color > 100 else "black"
            if piece_color == "white":
                white_pieces += 1
            else:
                black_pieces += 1

            bounding_boxes.append({
                "color": piece_color,
                "bbox": [int(x), int(y), int(w), int(h)]
            })

    return white_pieces, black_pieces, bounding_boxes, board_matrix.tolist()

def process_image(img_path):
    img = cv2.imread(img_path)
    preprocessed = preprocess_image(img)
    cells = get_grid_cells(preprocessed)
    white, black, boxes, matrix = detect_pieces(preprocessed, img, cells)
    return {
        "image_id": os.path.basename(img_path),
        "white_pieces": white,
        "black_pieces": black,
        "bounding_boxes": boxes,
        "board_matrix": matrix
    }

def process_images(input_folder, output_path):
    results = []
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(input_folder, filename)
            result = process_image(full_path)
            results.append(result)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    process_images(args.input_folder, args.output_json)
