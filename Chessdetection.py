import cv2
import numpy as np
import matplotlib.pyplot as plt 
import json
import math
import os

# Create output directory
output_dir = "processed_boards"          # to be removed after
os.makedirs(output_dir, exist_ok=True)

# Load image filenames from JSON file
with open("input.json", "r") as json_file:
    json_data = json.load(json_file)
    image_files = json_data["image_files"]

# Initialize the output results list
output_results = []

images = []

# Load all images from the filenames in the JSON
for filename in image_files:
    img = cv2.imread(filename)
    if img is not None:
        # Extract just the base filename without path
        base_filename = os.path.basename(filename)
        images.append((img, base_filename))
    else:
        print(f"Warning: Could not load image {filename}")

# Process all images
for idx, (image, filename) in enumerate(images):
    print(f"Processing image {idx+1}/{len(images)}: {filename}")
    # Convert to rgb for visualization
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Run Gaussian blur
    blurred = cv2.GaussianBlur(gray_image, (7, 7), 1.5)

    # Edge detection with Canny
    canny_image = cv2.Canny(blurred, 250, 180)      #low threshold of 250 and high of 180 - handpicked for the exact problem

    # Dilation to connect nearby edges
    kernel = np.ones((9, 9), np.uint8)      #using 9x9 kernel
    dilation_image = cv2.dilate(canny_image, kernel, iterations=1)

    # Hough Lines with adjusted parameters
    lines = cv2.HoughLinesP(dilation_image, 1, np.pi / 180, threshold=730, minLineLength=225, maxLineGap=110)       #Handpicked parameters also

    # Create an image that contains only black pixels
    black_image = np.zeros_like(dilation_image)

    # Group lines by similar angles to find dominant directions
    if lines is not None:
        # Group lines by angle function
        def group_lines_by_angle(lines, angle_threshold=5):
            angle_groups = {}
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate line length (longer lines have more weight)
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Calculate angle (0-180 degrees)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
                
                # Find if angle belongs to existing group
                found_group = False
                for group_angle in list(angle_groups.keys()):
                    if abs(angle - group_angle) < angle_threshold:
                        angle_groups[group_angle].append((line, length))
                        found_group = True
                        break
                        
                if not found_group:
                    angle_groups[angle] = [(line, length)]
            
            # Calculate total length of lines in each group
            group_lengths = {}
            for angle, lines_with_length in angle_groups.items():
                total_length = sum(length for _, length in lines_with_length)
                group_lengths[angle] = total_length
            
            # Sort groups by total line length (descending)
            sorted_groups = sorted(group_lengths.items(), key=lambda x: x[1], reverse=True)
            
            # Keep only the dominant groups (top 70% of total length)
            total_length = sum(group_lengths.values())
            threshold_length = 1 * total_length
            
            cumulative_length = 0
            dominant_angles = []
            for angle, length in sorted_groups:
                dominant_angles.append(angle)
                cumulative_length += length
                if cumulative_length >= threshold_length:
                    break
            
            # Get all lines from dominant groups
            filtered_lines = []
            for angle in dominant_angles:
                for line, _ in angle_groups[angle]:
                    filtered_lines.append(line)
            
            return filtered_lines
        
        # Apply filtering to keep only dominant line directions
        filtered_lines = group_lines_by_angle(lines)
        
        # Draw the filtered lines
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Apply a final dilation to connect any remaining gaps
        kernel = np.ones((3, 3), np.uint8)
        black_image = cv2.dilate(black_image, kernel, iterations=1)

        # Find contours
        board_contours, hierarchy = cv2.findContours(black_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Blank image for displaying valid contours (squares)
        valid_squares_image = np.zeros_like(black_image)

        # Improved square filtering
        valid_squares = []

        # Calculate image size for reference
        img_area = image.shape[0] * image.shape[1]

        # Fine-tuned parameters specifically for chess squares (it tries to detect mainly the 8x8 chess squares)
        min_square_area = 0.0007 * img_area
        max_square_area = 0.007 * img_area

        # Loop through contours and filter them by deciding if they are potential squares
        for contour in board_contours:
            area = cv2.contourArea(contour)
            if min_square_area < area < max_square_area:
                # Approximate the contour to a simpler shape
                epsilon = 0.03 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If polygon has 4 vertices (a quadrilateral)
                if len(approx) == 4:
                    # 4 points of polygon
                    pts = [pt[0].tolist() for pt in approx]

                    # Create same pattern for points
                    # Sort by x-coordinate (reverse for right-to-left order)
                    index_sorted = sorted(pts, key=lambda x: x[0], reverse=True)

                    # Adjust Y values to properly identify corners
                    if index_sorted[0][1] < index_sorted[1][1]:
                        cur = index_sorted[0]
                        index_sorted[0] = index_sorted[1]
                        index_sorted[1] = cur

                    if index_sorted[2][1] > index_sorted[3][1]:
                        cur = index_sorted[2]
                        index_sorted[2] = index_sorted[3]
                        index_sorted[3] = cur

                    # bottomright(1), topright(2), topleft(3), bottomleft(4)
                    pt1 = index_sorted[0]
                    pt2 = index_sorted[1]
                    pt3 = index_sorted[2]
                    pt4 = index_sorted[3]

                    # Calculate length of 4 sides of rectangle
                    l1 = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                    l2 = math.sqrt((pt2[0] - pt3[0])**2 + (pt2[1] - pt3[1])**2)
                    l3 = math.sqrt((pt3[0] - pt4[0])**2 + (pt3[1] - pt4[1])**2)
                    l4 = math.sqrt((pt1[0] - pt4[0])**2 + (pt1[1] - pt4[1])**2)
        
                    # Create a list of lengths
                    lengths = [l1, l2, l3, l4]
                    
                    # Get the maximum and minimum lengths
                    max_length = max(lengths)
                    min_length = min(lengths)

                    # Calculate the aspect ratio of the rectangle
                    aspect_ratio = max_length / min_length if min_length > 0 else float('inf')
                    
                    # Calculate diagonals
                    d1 = math.sqrt((pt1[0] - pt3[0])**2 + (pt1[1] - pt3[1])**2)
                    d2 = math.sqrt((pt2[0] - pt4[0])**2 + (pt2[1] - pt4[1])**2)
                    
                    # Calculate diagonal ratio
                    diagonal_ratio = max(d1, d2) / min(d1, d2) if min(d1, d2) > 0 else float('inf')
                    
                    # Fine-tuned thresholds for chess squares - Handpicked parameters to find all the possible squares in the grid
                    is_square = (
                        aspect_ratio < 2.7 and
                        diagonal_ratio < 2.7 and
                        (max_length - min_length) < 250
                    )
                    
                    if is_square:
                        valid_square = True
                        valid_squares.append(approx)

                        # Draw only valid squares to "valid_squares_image"
                        pt1_int = (int(pt1[0]), int(pt1[1]))
                        pt2_int = (int(pt2[0]), int(pt2[1]))
                        pt3_int = (int(pt3[0]), int(pt3[1]))
                        pt4_int = (int(pt4[0]), int(pt4[1]))
                        
                        cv2.line(valid_squares_image, pt1_int, pt2_int, (255, 255, 255), 15)
                        cv2.line(valid_squares_image, pt2_int, pt3_int, (255, 255, 255), 15)
                        cv2.line(valid_squares_image, pt3_int, pt4_int, (255, 255, 255), 15)
                        cv2.line(valid_squares_image, pt1_int, pt4_int, (255, 255, 255), 15)

        # Apply dilation to the valid_squares_image
        kernel = np.ones((7, 7), np.uint8)
        dilated_valid_squares_image = cv2.dilate(valid_squares_image, kernel, iterations=1)  

        # Create a convex hull mask to fill the chessboard
        contours, _ = cv2.findContours(dilated_valid_squares_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Create a new mask with the convex hull of the largest contour
            hull_mask = np.zeros_like(dilated_valid_squares_image)
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)
            cv2.drawContours(hull_mask, [hull], 0, 255, -1)
            
            # Apply the convex hull mask to get the final isolated chessboard
            filled_chessboard = cv2.bitwise_and(dilated_valid_squares_image, dilated_valid_squares_image, mask=hull_mask)
            
            # Update the chessboard mask and isolated image - chessboard_mask is now a rectangle filled with white so its easier to find
            chessboard_mask = hull_mask            
            dilated_valid_squares_image = filled_chessboard
        else:
            print(f"No contours found in the chessboard mask for {filename}")
            continue

        # Create a visualization image
        visual_img = cv2.cvtColor(chessboard_mask.copy(), cv2.COLOR_GRAY2BGR)

        # Find the contour of the chessboard (should be the largest contour)
        contours, _ = cv2.findContours(chessboard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"No contours found in {filename}")
            continue
        else:
            # Get the largest contour
            main_contour = max(contours, key=cv2.contourArea)
            
            # Try different epsilon values to get 4 points
            for epsilon_factor in [0.02, 0.015, 0.01, 0.005]:
                epsilon = epsilon_factor * cv2.arcLength(main_contour, True)
                approx = cv2.approxPolyDP(main_contour, epsilon, True)
                
                print(f"With epsilon factor {epsilon_factor}, found {len(approx)} points in {filename}")
                
                # If we have 4 points, perfect! 
                if len(approx) == 4:
                    break
                # If we have more than 4 points but not too many, we'll try to filter down to 4
                elif 4 < len(approx) <= 8:
                    break
            
            # If we have more than 4 points, we need to find the 4 most significant ones
            if len(approx) > 4:
                # We'll score points based on their angle
                points = [tuple(point[0]) for point in approx]
                
                # Find centroid of all points
                cx = sum(p[0] for p in points) / len(points)
                cy = sum(p[1] for p in points) / len(points)
                
                # Calculate angles between adjacent points
                angles = []
                for i in range(len(points)):
                    prev_idx = (i - 1) % len(points)
                    next_idx = (i + 1) % len(points)
                    
                    # Calculate vectors
                    v1 = (points[prev_idx][0] - points[i][0], points[prev_idx][1] - points[i][1])
                    v2 = (points[next_idx][0] - points[i][0], points[next_idx][1] - points[i][1])
                    
                    # Calculate angle between vectors
                    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                    v1_mag = math.sqrt(v1[0]**2 + v1[1]**2)
                    v2_mag = math.sqrt(v2[0]**2 + v2[1]**2)
                    
                    # Avoid division by zero
                    if v1_mag * v2_mag == 0:
                        cos_angle = 1.0  # Assume straight line (180 degrees) if magnitude is zero
                    else:
                        cos_angle = max(-1.0, min(1.0, dot_product / (v1_mag * v2_mag)))
                    
                    angle_rad = math.acos(cos_angle)
                    angle_deg = math.degrees(angle_rad)
                    
                    # Sharper angles (closer to 0 or 360) have higher scores
                    angle_score = 180 - abs(angle_deg - 180)
                    
                    # Also consider distance from centroid (farther is better for corners)
                    dist_from_center = math.sqrt((points[i][0] - cx)**2 + (points[i][1] - cy)**2)
                    
                    # Combined score favors sharp angles far from center
                    combined_score = angle_score + (0.01 * dist_from_center)
                    
                    angles.append((i, combined_score))
                
                # Sort by score (highest first)
                angles.sort(key=lambda x: x[1], reverse=True)
                
                # Take the indices of the top 4 corners
                best_indices = [idx for idx, _ in angles[:4]]
                best_indices.sort()  # Sort to maintain order around the contour
                
                # Extract the 4 best corners
                corners = [points[idx] for idx in best_indices]
            else:
                # We already have 4 or fewer points
                corners = [tuple(point[0]) for point in approx]
            
            # If we somehow ended up with fewer than 4 corners, fall back to convex hull approach
            if len(corners) < 4:
                print(f"Not enough corners found in {filename}, using convex hull method instead")
                # Find all white pixels
                white_pixels = np.where(chessboard_mask > 0)
                points = np.column_stack((white_pixels[1], white_pixels[0]))
                
                # Get the convex hull
                hull = cv2.convexHull(points.astype(np.float32))
                hull_points = [tuple(point[0]) for point in hull]
                
                # Use mathematical extremes to find corners
                corners = [
                    min(hull_points, key=lambda p: p[0] + p[1]),  # Min x+y
                    max(hull_points, key=lambda p: p[0] - p[1]),  # Max x-y
                    min(hull_points, key=lambda p: p[0] - p[1]),  # Min x-y
                    max(hull_points, key=lambda p: p[0] + p[1])   # Max x+y
                ]
            
            # Find the centroid of the corners
            center_x = sum(p[0] for p in corners) / len(corners)
            center_y = sum(p[1] for p in corners) / len(corners)
            
            # Sort corners by their angle from the centroid
            def get_angle(point):
                return math.atan2(point[1] - center_y, point[0] - center_x)
            
            # Sort corners clockwise
            corners.sort(key=get_angle)
            
            # Draw main contour
            cv2.drawContours(visual_img, [main_contour], 0, (0, 255, 255), 2)
            
            # Draw the center point
            cv2.circle(visual_img, (int(center_x), int(center_y)), 10, (255, 165, 0), -1)
            
            # Draw corners in the order they're found
            colors = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]  # Red, Green, Yellow, Blue
            for i, corner in enumerate(corners):
                x, y = int(corner[0]), int(corner[1])
                color = colors[i % len(colors)]
                
                # Draw the corner
                cv2.circle(visual_img, (x, y), 15, color, -1)
                
                # Add the corner number label
                cv2.putText(visual_img, f"C{i}", (x - 20, y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            # Draw the quadrilateral connecting the corners in their original order
            corners_array = np.array([(int(c[0]), int(c[1])) for c in corners]).reshape(-1, 1, 2)
            cv2.polylines(visual_img, [corners_array], True, (0, 255, 0), 3)
            
            # Create a dictionary to store results for this image
            image_results = {
                "image": filename,
                "num_pieces": 0,
                "board": [],
                "detected_pieces": []
            }
                        # Save the corners image
            plt.figure(figsize=(12, 8))
            plt.title(f"Corner Detection: {filename}")
            plt.imshow(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB))
            plt.savefig(os.path.join(output_dir, f"{idx+1}_{os.path.splitext(filename)[0]}_corners.png"))
            plt.close()
            
            # Print the corner coordinates
            for i, corner in enumerate(corners):
                print(f"Corner {i}: ({int(corner[0])}, {int(corner[1])}) - {filename}")
                
            # Make sure these are ordered correctly: [top_left, top_right, bottom_left, bottom_right]
            top_left = []
            top_right = []
            bottom_left = []
            bottom_right = []
            #assign every corner to the respective one
            for i, corner in enumerate(corners):
                if i == 0:
                    x, y = int(corner[0]), int(corner[1])
                    top_left.append(x)
                    top_left.append(y)
                if i == 1:
                    x, y = int(corner[0]), int(corner[1])
                    top_right.append(x)
                    top_right.append(y)
                if i == 2:
                    x, y = int(corner[0]), int(corner[1])
                    bottom_right.append(x)
                    bottom_right.append(y)
                if i == 3:
                    x, y = int(corner[0]), int(corner[1])
                    bottom_left.append(x)
                    bottom_left.append(y)

            extreme_points_list = np.float32([
                [top_left[0], top_left[1]],
                [top_right[0], top_right[1]],
                [bottom_left[0], bottom_left[1]],
                [bottom_right[0], bottom_right[1]]
            ])

            # Set the output image dimensions
            threshold = 0  # Extra space on all sides
            width, height = 1200, 1200

            # Define the destination points (shifted by 'threshold' on all sides)
            dst_pts = np.float32([
                [threshold, threshold],  # Top-left
                [width + threshold, threshold],  # Top-right
                [threshold, height + threshold],  # Bottom-left
                [width + threshold, height + threshold]  # Bottom-right
            ])

            # Compute the perspective transform matrix
            M = cv2.getPerspectiveTransform(extreme_points_list, dst_pts)

            # Apply the transformation with extra width and height
            warped_image = cv2.warpPerspective(image, M, (width + 2 * threshold, height + 2 * threshold))

            # Mark the corners in the warped image for verification
            cv2.circle(warped_image, (threshold, threshold), 15, (0, 0, 255), -1)  # Top-left
            cv2.circle(warped_image, (width + threshold, threshold), 15, (0, 255, 0), -1)  # Top-right
            cv2.circle(warped_image, (threshold, height + threshold), 15, (255, 255, 0), -1)  # Bottom-left
            cv2.circle(warped_image, (width + threshold, height + threshold), 15, (255, 0, 0), -1)  # Bottom-right

            # Next, divide the board into 64 squares
            rows, cols = 8, 8
            square_width = width // cols
            square_height = height // rows

            # Create a list to store all 64 squares with their coordinates
            squares_data = []

            # Extract each square's coordinates in the warped image
            # Go from bottom to top (rank 1 to 8) and left to right (file a to h)
            for i in range(rows - 1, -1, -1):  # Start from bottom row (rank 1)
                for j in range(cols):  # Left to right (file a to h)
                    # Define the 4 corners of each square
                    top_left_square = (j * square_width, i * square_height)
                    top_right_square = ((j + 1) * square_width, i * square_height)
                    bottom_left_square = (j * square_width, (i + 1) * square_height)
                    bottom_right_square = ((j + 1) * square_width, (i + 1) * square_height)
                    
                    # Calculate center of the square
                    center_x = (top_left_square[0] + bottom_right_square[0]) // 2
                    center_y = (top_left_square[1] + bottom_right_square[1]) // 2
                    
                    # Append to the list
                    squares_data.append({
                        "center": (center_x, center_y),
                        "corners": [bottom_right_square, top_right_square, top_left_square, bottom_left_square],
                        "rank": 8 - i,  # Chess rank (1-8)
                        "file": chr(97 + j)  # Chess file (a-h)
                    })

            # For visualization, create a copy of the warped image
            square_centers_image = warped_image.copy()

            # Draw square centers and labels
            for square in squares_data:
                center = square["center"]
                rank = square["rank"]
                file = square["file"]
                
                # Draw center point
                cv2.circle(square_centers_image, center, 5, (255, 0, 0), -1)
                
                # Add coordinate label
                label = f"{file}{rank}"
                cv2.putText(square_centers_image, label, (center[0] - 20, center[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Save the square centers image - for
            plt.figure(figsize=(12, 10))
            plt.imshow(cv2.cvtColor(square_centers_image, cv2.COLOR_BGR2RGB))
            plt.title("Chessboard with Square Centers and Coordinates")
            plt.savefig(os.path.join(output_dir, f"{idx+1}_{os.path.splitext(filename)[0]}_square_centers.png"))
            plt.close()

            # Now we need to map these warped coordinates back to the original image
            # Convert square centers to numpy array for transformation
            square_centers_warped = np.array([square["center"] for square in squares_data], dtype=np.float32).reshape(-1, 1, 2)

            # Compute the inverse perspective transformation matrix
            M_inv = cv2.invert(M)[1]

            # Transform centers back to original image coordinates
            square_centers_original = cv2.perspectiveTransform(square_centers_warped, M_inv)

            # Update the squares_data with original image coordinates
            for i, square in enumerate(squares_data):
                square["original_center"] = (
                    int(square_centers_original[i][0][0]),
                    int(square_centers_original[i][0][1])
                )

            # Visualize the square centers on the original image
            original_with_centers = image.copy()

            for square in squares_data:
                center = square["original_center"]
                rank = square["rank"]
                file = square["file"]
                
                # Draw center point
                cv2.circle(original_with_centers, center, 8, (0, 255, 0), -1)
                
                # Add coordinate label
                label = f"{file}{rank}"
                cv2.putText(original_with_centers, label, (center[0] - 15, center[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Simplified Chess Piece Detection - Only Detecting Piece Presence
            def detect_pieces_on_warped_image_rgb_validated(warped_image, squares_data):
                visualization = warped_image.copy()
                gray_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

                roi_size = 35
                initial_board_matrix = [[0 for _ in range(8)] for _ in range(8)]
                empty_light_rgbs = []
                empty_dark_rgbs = []
                detected_pieces = []  # List to store detected piece bounding boxes

                # === Stage 1: Traditional heuristic ===
                for square in squares_data:
                    center = square["center"]
                    file = square["file"]
                    rank = square["rank"]
                    file_idx = ord(file) - ord('a')
                    rank_idx = 8 - rank
                    is_dark_square = (file_idx + rank_idx) % 2 == 1

                    roi_x = max(0, center[0] - roi_size)
                    roi_y = max(0, center[1] - roi_size)
                    roi_width = min(roi_size * 2, warped_image.shape[1] - roi_x)
                    roi_height = min(roi_size * 2, warped_image.shape[0] - roi_y)

                    roi_gray = gray_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
                    if roi_gray.size == 0:
                        continue

                    std_dev = np.std(roi_gray)
                    edges = cv2.Canny(roi_gray, 35, 150)
                    edge_density = np.count_nonzero(edges) / roi_gray.size

                    if std_dev > 15 or edge_density > 0.07:  # was 18 and 0.08
                        initial_board_matrix[rank_idx][file_idx] = 1  # mark as piece

                # === Stage 2: Average RGB of empty squares ===
                for square in squares_data:
                    center = square["center"]
                    file = square["file"]
                    rank = square["rank"]
                    file_idx = ord(file) - ord('a')
                    rank_idx = 8 - rank
                    is_dark_square = (file_idx + rank_idx) % 2 == 1

                    if initial_board_matrix[rank_idx][file_idx] == 1:
                        continue  # square is occupied

                    roi_x = max(0, center[0] - roi_size)
                    roi_y = max(0, center[1] - roi_size)
                    roi_width = min(roi_size * 2, warped_image.shape[1] - roi_x)
                    roi_height = min(roi_size * 2, warped_image.shape[0] - roi_y)
                    roi = warped_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

                    if roi.size == 0:
                        continue

                    mean_rgb = np.mean(roi.reshape(-1, 3), axis=0)

                    if is_dark_square:
                        empty_dark_rgbs.append(mean_rgb)
                    else:
                        empty_light_rgbs.append(mean_rgb)

                mean_rgb_light = np.mean(empty_light_rgbs, axis=0) if empty_light_rgbs else np.array([200, 200, 200])
                mean_rgb_dark = np.mean(empty_dark_rgbs, axis=0) if empty_dark_rgbs else np.array([80, 80, 80])

                print("Estimated average RGB for empty light squares:", mean_rgb_light)
                print("Estimated average RGB for empty dark squares:", mean_rgb_dark)

                # === Stage 3: RGB Detection ===
                threshold_rgb_diff = 30  # sensitivity (was 37)
                board_matrix = [[0 for _ in range(8)] for _ in range(8)]

                for square in squares_data:
                    center = square["center"]
                    file = square["file"]
                    rank = square["rank"]
                    file_idx = ord(file) - ord('a')
                    rank_idx = 8 - rank
                    is_dark_square = (file_idx + rank_idx) % 2 == 1
                    expected_rgb = mean_rgb_dark if is_dark_square else mean_rgb_light

                    roi_x = max(0, center[0] - roi_size)
                    roi_y = max(0, center[1] - roi_size)
                    roi_width = min(roi_size * 2, warped_image.shape[1] - roi_x)
                    roi_height = min(roi_size * 2, warped_image.shape[0] - roi_y)
                    roi = warped_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

                    if roi.size == 0:
                        continue

                    mean_rgb = np.mean(roi.reshape(-1, 3), axis=0)
                    diff = np.linalg.norm(mean_rgb - expected_rgb)

                    if diff > threshold_rgb_diff:
                        # === Stage 4: validation with heuristic ===
                        roi_gray = gray_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
                        std_dev = np.std(roi_gray)
                        edges = cv2.Canny(roi_gray, 30, 130)  # was 35, 150
                        edge_density = np.count_nonzero(edges) / roi_gray.size

                        std_threshold = 13 if is_dark_square else 9
                        edge_threshold = 0.045 if is_dark_square else 0.025

                        if std_dev > std_threshold or edge_density > edge_threshold:
                            # Update board matrix to mark piece presence
                            board_matrix[rank_idx][file_idx] = 1
                            
                            # Get original image coordinates for the square's center
                            original_center = square["original_center"]
                            
                            # Create bounding box around the original center
                            original_roi_size = 40  # Adjust as needed
                            original_roi_x = max(0, original_center[0] - original_roi_size)
                            original_roi_y = max(0, original_center[1] - original_roi_size)
                            original_roi_width = 2 * original_roi_size
                            original_roi_height = 2 * original_roi_size
                            
                            # Add detected piece to the list
                            detected_piece = {
                                "xmin": original_roi_x,
                                "ymin": original_roi_y,
                                "xmax": original_roi_x + original_roi_width,
                                "ymax": original_roi_y + original_roi_height
                            }
                            detected_pieces.append(detected_piece)
                            
                            # Draw visualization on the warped image
                            cv2.rectangle(visualization,
                                        (roi_x, roi_y),
                                        (roi_x + roi_width, roi_y + roi_height),
                                        (0, 255, 0), 2)
                            cv2.putText(visualization, f"{file}{rank}",
                                        (center[0] - 15, center[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # Update image_results with board matrix, piece count, and detected pieces
                image_results["board"] = board_matrix
                image_results["num_pieces"] = sum(sum(row) for row in board_matrix)
                image_results["detected_pieces"] = detected_pieces

                return visualization


            # Run piece detection on the warped image instead of original
            warped_piece_detection_img = detect_pieces_on_warped_image_rgb_validated(warped_image, squares_data)
            # Save the piece detection image
            plt.figure(figsize=(12, 10))
            plt.imshow(cv2.cvtColor(warped_piece_detection_img, cv2.COLOR_BGR2RGB))
            plt.title("Chess Piece Detection (Presence Only)")
            plt.savefig(os.path.join(output_dir, f"{idx+1}_{os.path.splitext(filename)[0]}_piece_detection.png"))
            plt.close()
            
            # Add results to output list
            output_results.append(image_results)

if __name__ == "__main__":
    # Write results to output.json
    with open("output.json", "w") as json_file:
        json.dump(output_results, json_file, indent=4)

    print(f"Processing complete. Results saved to output.json")
    print(f"Output visualizations saved to {output_dir}/")
