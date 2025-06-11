import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from ultralytics import YOLO

def chesspos2number(chesspos):
    col = ord(chesspos[0]) - ord('a')
    row = int(chesspos[1]) - 1
    return row, col

def get_piece_symbol(piece_name):
    piece_symbols = {
        'white-king': '♚', 'white-queen': '♛', 'white-rook': '♜',
        'white-bishop': '♝', 'white-knight': '♞', 'white-pawn': '♟',
        'black-king': '♚', 'black-queen': '♛', 'black-rook': '♜',
        'black-bishop': '♝', 'black-knight': '♞', 'black-pawn': '♟'
    }
    return piece_symbols.get(piece_name, '?')

def find_image_id_by_filename(filename, annotations):
    for img in annotations['images']:
        if filename in img['file_name']:
            return img['id']
    return None

def load_corners_from_dataset(image_id, annotations):
    for corner_ann in annotations['annotations']['corners']:
        if corner_ann['image_id'] == image_id:
            corners_dict = corner_ann['corners']
            return np.array([
                corners_dict['top_left'],
                corners_dict['top_right'],
                corners_dict['bottom_right'],
                corners_dict['bottom_left']
            ], dtype=np.float32)
    return None

def get_visual_corner_mapping(corners):
    corners_with_indices = [(i, corner) for i, corner in enumerate(corners)]
    sorted_by_y = sorted(corners_with_indices, key=lambda x: x[1][1])
    top_corners = sorted_by_y[:2]
    bottom_corners = sorted_by_y[2:]
    top_sorted = sorted(top_corners, key=lambda x: x[1][0])
    bottom_sorted = sorted(bottom_corners, key=lambda x: x[1][0])
    
    visual_order = [top_sorted[0], top_sorted[1], bottom_sorted[1], bottom_sorted[0]]
    annotation_labels = ['TL', 'TR', 'BR', 'BL']
    display_positions = [(0, 0), (7, 0), (7, 7), (0, 7)]
    
    annotation_to_display = {}
    for i, (orig_idx, corner) in enumerate(visual_order):
        annotation_name = annotation_labels[orig_idx]
        annotation_to_display[annotation_name] = display_positions[i]
    
    return annotation_to_display, [corner for _, corner in visual_order]

def pixel_to_chess_square(x, y, corners, board_size=800):
    _, visual_corners = get_visual_corner_mapping(corners)
    visual_corners = np.array(visual_corners, dtype=np.float32)
    
    dst_corners = np.array([
        [0, 0], [board_size, 0], [board_size, board_size], [0, board_size]
    ], dtype=np.float32)
    
    transform_matrix = cv2.getPerspectiveTransform(visual_corners, dst_corners)
    point = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, transform_matrix)
    board_x, board_y = transformed[0][0]
    
    if board_x < 0 or board_x > board_size or board_y < 0 or board_y > board_size:
        return None
    
    col = max(0, min(7, int(board_x // (board_size / 8))))
    row = max(0, min(7, int(board_y // (board_size / 8))))
    
    chess_row = 8 - row
    chess_col = chr(ord('a') + col)
    return f"{chess_col}{chess_row}"

def create_virtual_board(detections, corners, class_names):
    board = np.zeros((8, 8), dtype=int)
    piece_info = {}
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        base_x = (x1 + x2) / 2
        base_y = y2
        
        chess_square = pixel_to_chess_square(base_x, base_y, corners)
        if chess_square is None:
            continue
        
        row, col = chesspos2number(chess_square)
        piece_name = class_names[detection['class']]
        confidence = detection['confidence']
        
        if chess_square in piece_info:
            if confidence > piece_info[chess_square]['confidence']:
                board[row][col] = 1
                piece_info[chess_square] = {'name': piece_name, 'confidence': confidence}
        else:
            board[row][col] = 1
            piece_info[chess_square] = {'name': piece_name, 'confidence': confidence}
    
    return board, piece_info

def create_visualization(result):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    ax1.imshow(result['image'])
    ax1.set_title("YOLO Detections")
    
    for detection in result['detections']:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor='red', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x1, y1-5, f'{confidence:.2f}', 
                color='red', fontsize=10, weight='bold')
    
    corners = result['corners']
    corners_plot = np.vstack([corners, corners[0]])
    ax1.plot(corners_plot[:, 0], corners_plot[:, 1], 'b-', linewidth=3)
    
    corner_labels = ['TL', 'TR', 'BR', 'BL']
    corner_colors = ['red', 'green', 'blue', 'orange']
    for corner, label, color in zip(corners, corner_labels, corner_colors):
        ax1.plot(corner[0], corner[1], 'o', color=color, markersize=10)
        ax1.text(corner[0], corner[1], label, fontsize=12, color='white', weight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
    ax1.axis('off')
    
    annotation_to_display, _ = get_visual_corner_mapping(corners)
    tl_pos = annotation_to_display['TL']
    
    chessboard = np.zeros((8, 8, 3))
    for i in range(8):
        for j in range(8):
            offset_from_tl = (i - tl_pos[1]) + (j - tl_pos[0])
            if offset_from_tl % 2 == 0:
                chessboard[i, j] = [0.94, 0.89, 0.76]
            else:
                chessboard[i, j] = [0.72, 0.53, 0.35]
    
    ax2.imshow(chessboard)
    ax2.set_title("Virtual Board")
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    corner_offsets = {
        'TL': (-0.4, -0.4), 'TR': (0.4, -0.4),
        'BR': (0.4, 0.4), 'BL': (-0.4, 0.4)
    }
    
    for annotation_name, pos in annotation_to_display.items():
        color = corner_colors[corner_labels.index(annotation_name)]
        offset_x, offset_y = corner_offsets[annotation_name]
        corner_x = pos[0] + offset_x
        corner_y = pos[1] + offset_y
        
        ax2.plot(corner_x, corner_y, 'o', color=color, markersize=6)
        ax2.text(corner_x, corner_y, annotation_name, fontsize=8, color='white', weight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.1", facecolor=color, alpha=0.9))
    
    for square, info in result['piece_info'].items():
        row, col = chesspos2number(square)
        display_row = 7 - row
        piece_name = info['name']
        piece_symbol = get_piece_symbol(piece_name)
        piece_color = 'white' if 'white' in piece_name else 'black'
        
        ax2.text(col, display_row, piece_symbol, 
                ha='center', va='center', fontsize=50, color=piece_color, weight='bold')
    
    plt.tight_layout()
    return fig

def process_image(image_path, image_id, model, annotations):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    corners = load_corners_from_dataset(image_id, annotations)
    if corners is None:
        return None
    
    results = model(image_path, conf=0.25)
    result = results[0]
    
    detections = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            detections.append({
                'bbox': box,
                'class': class_id,
                'confidence': conf
            })
    
    board_matrix, piece_info = create_virtual_board(detections, corners, model.names)
    
    return {
        'image': image_rgb,
        'corners': corners,
        'detections': detections,
        'board_matrix': board_matrix,
        'piece_info': piece_info
    }

def detect_chess_pieces(image_path, model_path='yolo_chess_model.pt', annotations_path='annotations.json'):
    """
    Main function to detect chess pieces and return visualization figure.
    
    Args:
        image_path (str): Path to the chess board image
        model_path (str): Path to YOLO model file
        annotations_path (str): Path to annotations JSON file
    
    Returns:
        matplotlib.figure.Figure: Figure with chess detection visualization
        dict: Detection results with piece information
    """
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    image_filename = image_path.split('/')[-1].split('\\')[-1]
    image_id = find_image_id_by_filename(image_filename, annotations)
    
    if image_id is None:
        raise ValueError(f"Image {image_filename} not found in annotations")
    
    model = YOLO(model_path)
    result = process_image(image_path, image_id, model, annotations)
    
    if result is None:
        raise ValueError("Failed to process image")
    
    fig = create_visualization(result)
    
    return fig, result

if __name__ == "__main__":
    image_path = 'yolo_ds/images/test/G000_IMG090.jpg'
    
    try:
        fig, result = detect_chess_pieces(image_path)
        plt.show()
        
        print(f"Detected {len(result['piece_info'])} pieces:")
        for square, info in sorted(result['piece_info'].items()):
            print(f"  {square}: {info['name']} ({info['confidence']:.3f})")
    except Exception as e:
        print(f"Error: {e}")