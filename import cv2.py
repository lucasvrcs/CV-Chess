import cv2
import numpy as np
import json
import os
from matplotlib import pyplot as plt

def process_chess_image(image_path):
    """
    Process a chess board image and detect pieces.
    
    Args:
        image_path: Path to the chess board image
        
    Returns:
        Dictionary with results (piece counts, positions, board matrix)
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    
    # Convert to RGB for visualization (OpenCV loads as BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Preprocessing
    # Convert to grayscale for board detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Chess board detection
    # Using Canny edge detection to find edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (likely the chess board)
    board_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to get a simplified polygon
    epsilon = 0.02 * cv2.arcLength(board_contour, True)
    approx_board = cv2.approxPolyDP(board_contour, epsilon, True)
    
    # If we have a quadrilateral (4 points), we can perform perspective transform
    if len(approx_board) == 4:
        # Order the points [top-left, top-right, bottom-right, bottom-left]
        pts = np.array([pt[0] for pt in approx_board], dtype="float32")
        
        # Define a function to order points
        def order_points(pts):
            # Initialize a list of coordinates that will be ordered
            rect = np.zeros((4, 2), dtype="float32")
            
            # Top-left point will have the smallest sum
            # Bottom-right point will have the largest sum
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            
            # Top-right point will have the smallest difference
            # Bottom-left will have the largest difference
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            return rect
        
        # Get ordered points
        rect = order_points(pts)
        
        # Define destination points for perspective transform (800x800 square)
        width, height = 800, 800
        dst = np.array([
            [0, 0],           # Top-left
            [width - 1, 0],   # Top-right
            [width - 1, height - 1], # Bottom-right
            [0, height - 1]   # Bottom-left
        ], dtype="float32")
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply perspective transform
        warped = cv2.warpPerspective(img, M, (width, height))
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # 4. Detect chess squares
        # We know that the board is 8x8, so each square is width/8 by height/8
        square_size = width // 8
        
        # Create empty board matrix (8x8)
        board_matrix = np.zeros((8, 8), dtype=int)
        
        # Lists to store detected pieces
        white_pieces_list = []
        black_pieces_list = []
        pieces_positions = []
        
        # Process each square
        for row in range(8):
            for col in range(8):
                # Get the square region
                y1 = row * square_size
                y2 = (row + 1) * square_size
                x1 = col * square_size
                x2 = (col + 1) * square_size
                
                square = warped[y1:y2, x1:x2]
                
                # Convert square to HSV for better color discrimination
                square_hsv = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)
                
                # Get square mask (ignore dark and light checkerboard colors)
                # Focus on pieces by thresholding in HSV space
                
                # Mask for white pieces (high value in HSV)
                white_mask = cv2.inRange(square_hsv, 
                                         np.array([0, 0, 180]), 
                                         np.array([180, 80, 255]))
                
                # Mask for black pieces (low value in HSV)
                black_mask = cv2.inRange(square_hsv, 
                                        np.array([0, 0, 0]), 
                                        np.array([180, 255, 60]))
                
                # Check if there might be a piece in this square
                # Count white and black pixels and check if they exceed thresholds
                white_pixel_count = cv2.countNonZero(white_mask)
                black_pixel_count = cv2.countNonZero(black_mask)
                
                # Thresholds for piece detection (adjust as needed)
                piece_threshold = (square_size * square_size) * 0.1  # 10% of square area
                
                has_piece = False
                piece_color = None
                
                if white_pixel_count > piece_threshold:
                    # Likely a white piece
                    has_piece = True
                    piece_color = "white"
                    white_pieces_list.append((row, col))
                elif black_pixel_count > piece_threshold:
                    # Likely a black piece
                    has_piece = True
                    piece_color = "black"
                    black_pieces_list.append((row, col))
                
                if has_piece:
                    # Update board matrix
                    board_matrix[row, col] = 1
                    
                    # Calculate bounding box in original image coordinates
                    # We need to map back from the warped coordinates to original
                    # This is an approximation since we're using the center of the square
                    center_x = x1 + square_size // 2
                    center_y = y1 + square_size // 2
                    
                    # Map center point back to original image
                    # This uses the inverse perspective transform
                    center_warped = np.array([[center_x, center_y]], dtype=np.float32)
                    center_orig = cv2.perspectiveTransform(
                        center_warped.reshape(-1, 1, 2), 
                        cv2.invert(M)[1]
                    ).reshape(-1, 2)[0]
                    
                    # Create a box around the center point
                    box_size = 50  # Approximate size in original image
                    x_min = int(center_orig[0] - box_size // 2)
                    y_min = int(center_orig[1] - box_size // 2)
                    x_max = int(center_orig[0] + box_size // 2)
                    y_max = int(center_orig[1] + box_size // 2)
                    
                    # Add bounding box to pieces_positions
                    pieces_positions.append({
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "color": piece_color
                    })
        
        # 5. Visualization and results
        # Draw detected pieces on a clean warped board image for visualization
        viz_board = warped_rgb.copy()
        for row, col in white_pieces_list:
            y1 = row * square_size
            y2 = (row + 1) * square_size
            x1 = col * square_size
            x2 = (col + 1) * square_size
            cv2.rectangle(viz_board, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(viz_board, "W", (x1 + 10, y1 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        for row, col in black_pieces_list:
            y1 = row * square_size
            y2 = (row + 1) * square_size
            x1 = col * square_size
            x2 = (col + 1) * square_size
            cv2.rectangle(viz_board, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(viz_board, "B", (x1 + 10, y1 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Visualize the results
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1), plt.imshow(img_rgb), plt.title('Original Image')
        plt.subplot(2, 3, 2), plt.imshow(edges, cmap='gray'), plt.title('Edges')
        plt.subplot(2, 3, 3), plt.imshow(warped_rgb), plt.title('Warped Board')
        plt.subplot(2, 3, 4), plt.imshow(viz_board), plt.title('Detected Pieces')
        
        # Draw original image with bounding boxes
        img_with_boxes = img_rgb.copy()
        for pos in pieces_positions:
            color = (0, 255, 0) if pos["color"] == "white" else (255, 0, 0)
            cv2.rectangle(img_with_boxes, 
                          (pos["x_min"], pos["y_min"]), 
                          (pos["x_max"], pos["y_max"]), 
                          color, 2)
        
        plt.subplot(2, 3, 5), plt.imshow(img_with_boxes), plt.title('Boxes on Original')
        
        # Visualize the board matrix
        plt.subplot(2, 3, 6)
        plt.imshow(board_matrix, cmap='binary')
        plt.title('Board Matrix')
        for i in range(8):
            for j in range(8):
                plt.text(j, i, str(board_matrix[i, j]), 
                        ha="center", va="center", color="red")
        
        plt.tight_layout()
        plt.show()
        
        # Return results
        results = {
            "white_pieces": len(white_pieces_list),
            "black_pieces": len(black_pieces_list),
            "pieces_positions": pieces_positions,
            "board_matrix": board_matrix.tolist()
        }
        
        return results
    
    else:
        # If we don't find a good board contour, use a simplified approach
        print(f"Could not detect a quadrilateral board, found {len(approx_board)} points")
        
        # Visualization
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1), plt.imshow(img_rgb), plt.title('Original Image')
        plt.subplot(2, 2, 2), plt.imshow(gray, cmap='gray'), plt.title('Grayscale')
        plt.subplot(2, 2, 3), plt.imshow(edges, cmap='gray'), plt.title('Edges')
        
        # Draw detected contour
        contour_img = img_rgb.copy()
        cv2.drawContours(contour_img, [board_contour], -1, (0, 255, 0), 3)
        plt.subplot(2, 2, 4), plt.imshow(contour_img), plt.title('Detected Contour')
        
        plt.tight_layout()
        plt.show()
        
        # Return empty results
        results = {
            "white_pieces": 0,
            "black_pieces": 0,
            "pieces_positions": [],
            "board_matrix": np.zeros((8, 8), dtype=int).tolist()
        }
        
        return results

def process_dataset(images_folder, output_file):
    """
    Process all PNG images in the specified folder and write results to output JSON.
    
    Args:
        images_folder: Path to folder containing PNG images
        output_file: Path to output JSON file for results
    """
    results = []
    
    # Get all PNG files in the folder
    image_files = [f for f in os.listdir(images_folder) 
                 if f.lower().endswith('.jpg')]
    
    # Process each image
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(images_folder, img_file)
        
        print(f"Processing image {i+1}/{len(image_files)}: {img_file}")
        
        # Process the image
        img_results = process_chess_image(img_path)
        
        if img_results:
            # Add to results list
            results.append({
                "id": img_file,  # Using filename as ID
                "white_pieces": img_results["white_pieces"],
                "black_pieces": img_results["black_pieces"],
                "pieces_positions": img_results["pieces_positions"],
                "board_matrix": img_results["board_matrix"]
            })
    
    # Write output JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} images. Results saved to {output_file}")

# For testing with a single image
def test_single_image(image_path):
    results = process_chess_image(image_path)
    if results:
        print("Results:")
        print(f"- White pieces: {results['white_pieces']}")
        print(f"- Black pieces: {results['black_pieces']}")
        print(f"- Number of pieces detected: {len(results['pieces_positions'])}")
        print(f"- Board matrix: \n{np.array(results['board_matrix'])}")
    
    return results

# Main execution
if __name__ == "__main__":
    # For testing a single image:
    # test_single_image("images/sample_image.png")
    
    # For processing the full dataset:
    # process_dataset("images", "chess_results.json")
    
    # For now, let's just test with a specific image if one is provided
    import sys
    if len(sys.argv) > 1:
        test_single_image(sys.argv[1])
    else:
        # Default: process all images in the images folder
        images_folder = "images"
        if os.path.exists(images_folder):
            process_dataset(images_folder, "chess_results.json")
        else:
            print(f"Images folder '{images_folder}' not found. Please create it and add PNG images.")