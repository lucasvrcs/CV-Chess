import cv2
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import  math
import csv
from reportlab.graphics import renderPM
from PIL import Image
import os



image_path = r"images" 
images = []

# Get all image filenames in the folder
for filename in sorted(os.listdir(image_path)):
    if filename.endswith(".jpg"): 
        img = cv2.imread(os.path.join(image_path, filename))
        if img is not None:
            images.append(img)

images = images[0:2]
for idx, image in enumerate(images):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(9, 7))
    plt.imshow(rgb_image)
    plt.title(f"RGB Image {idx+1}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()  # <-- this is required to actually display the image
    



    ##########################################################




    blurred = cv2.GaussianBlur(gray_image, (3, 3), 1.5)#added
    ret, otsu_binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to clean up the image
    kernel_open = np.ones((5, 5), np.uint8)
    morph_image = cv2.morphologyEx(otsu_binary, cv2.MORPH_OPEN, kernel_open)
    morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_CLOSE, kernel_open)
    # blurred = cv2.GaussianBlur(morph_image, (5, 5), 1.5) #

    # Canny edge detection
    canny_image = cv2.Canny(otsu_binary, 30, 255)               # ta na otsu binary caguei no morph


    # Dilation to connect nearby edges
    kernel = np.ones((9, 9), np.uint8)              # mudar isto might also cook ig
    dilation_image = cv2.dilate(canny_image, kernel, iterations=1)

    plt.figure(figsize=(9, 7))
    plt.title("Grid Lines Overlay")
    plt.imshow(dilation_image)
    plt.show()

    # Hough Lines with adjusted parameters
    # Higher threshold to detect only stronger lines
    # Higher minLineLength to favor longer lines (grid lines span the board)
    # Lower maxLineGap to avoid connecting unrelated segments

    #lines = cv2.HoughLinesP(dilation_image, 1, np.pi / 180, threshold=550, minLineLength=125, maxLineGap=120) #120
    lines = cv2.HoughLinesP(dilation_image, 1, np.pi / 180, threshold=550, minLineLength=225, maxLineGap=110) #120

    # Create an image that contains only black pixels
    black_image = np.zeros_like(dilation_image)

    # Group lines by similar angles to find dominant directions
    if lines is not None:
        # Group lines by angle
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

        # Display the result
        plt.figure(figsize=(9, 7))
        plt.title("Filtered Grid Lines")
        plt.imshow(black_image, cmap="gray")
        plt.show()

        # Also show the grid lines overlaid on the original image
        overlay_image = rgb_image.copy()

        # Create a mask from the black_image
        grid_mask = black_image > 0

        # Apply grid lines to the overlay image
        overlay_image[grid_mask] = [0, 255, 0]  # Green color for grid lines

        # Display the overlay
        plt.figure(figsize=(9, 7))
        plt.title("Grid Lines Overlay")
        plt.imshow(overlay_image)
        plt.show()


        ############################################################

        # Look for valid squares and check if squares are inside of board

        # Find contours
        board_contours, hierarchy = cv2.findContours(black_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Blank image for displaying all contours
        all_contours_image = np.zeros_like(black_image)

        # Copy input image for displaying all squares 
        squares_image = np.copy(image) 

        # Blank image for displaying valid contours (squares)
        valid_squares_image = np.zeros_like(black_image)

        # Improved square filtering
        valid_squares = []

        # Calculate image size for reference
        img_area = image.shape[0] * image.shape[1]

        # Fine-tuned parameters specifically for chess squares
        # Chess squares should be within a certain size range relative to the image
        min_square_area = 0.0007 * img_area  # Adjust based on your chess board size
        max_square_area = 0.007 * img_area   # Adjust based on your chess board size

        # For the edge case with smaller squares, you might need to adjust:
        # min_square_area = 0.0005 * img_area
        # max_square_area = 0.01 * img_area

        # Print parameters for debugging
        print(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
        print(f"Min square area: {min_square_area:.2f}, Max square area: {max_square_area:.2f}")
        print(f"Total contours found: {len(board_contours)}")

        # Loop through contours and filter them by deciding if they are potential squares
        for contour in board_contours:
            area = cv2.contourArea(contour)
            if min_square_area < area < max_square_area:
                # Approximate the contour to a simpler shape
                epsilon = 0.03 * cv2.arcLength(contour, True)  # Balanced epsilon value #mudei de 0.03
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
                    
                    # Chess squares should be square-ish, with reasonable ratios
                    # Fine-tuned thresholds for chess squares
                    is_square = (
                        aspect_ratio < 2.2 and       # Sides should be similar (square-like) #mudei para 1
                        diagonal_ratio < 2.2 and     # Diagonals should be similar
                        (max_length - min_length) < 250  # Max difference between sides
                    )
                    
                    if is_square:
                        valid_square = True
                        valid_squares.append(approx)
                    else:
                        valid_square = False
        
                    # Draw all quadrilaterals to "all_contours_image"
                    # Use integer coordinates to avoid drawing errors
                    pt1_int = (int(pt1[0]), int(pt1[1]))
                    pt2_int = (int(pt2[0]), int(pt2[1]))
                    pt3_int = (int(pt3[0]), int(pt3[1]))
                    pt4_int = (int(pt4[0]), int(pt4[1]))
                    
                    cv2.line(all_contours_image, pt1_int, pt2_int, (255, 255, 255), 1)
                    cv2.line(all_contours_image, pt2_int, pt3_int, (255, 255, 255), 1)
                    cv2.line(all_contours_image, pt3_int, pt4_int, (255, 255, 255), 1)
                    cv2.line(all_contours_image, pt1_int, pt4_int, (255, 255, 255), 1)
        
                    if valid_square:
                        # Draw the lines between the points for valid squares
                        cv2.line(squares_image, pt1_int, pt2_int, (0, 255, 255), 2)  # Yellow
                        cv2.line(squares_image, pt2_int, pt3_int, (0, 255, 255), 2)
                        cv2.line(squares_image, pt3_int, pt4_int, (0, 255, 255), 2)
                        cv2.line(squares_image, pt1_int, pt4_int, (0, 255, 255), 2)

                        # Draw only valid squares to "valid_squares_image"
                        cv2.line(valid_squares_image, pt1_int, pt2_int, (255, 255, 255), 15)
                        cv2.line(valid_squares_image, pt2_int, pt3_int, (255, 255, 255), 15)
                        cv2.line(valid_squares_image, pt3_int, pt4_int, (255, 255, 255), 15)
                        cv2.line(valid_squares_image, pt1_int, pt4_int, (255, 255, 255), 15)

        print(f"Found {len(valid_squares)} valid squares")

        # Display results
        plt.figure(figsize=(18, 15))

        plt.subplot(131)
        plt.title("Geometrically valid squares on original image\nsquares_image")
        plt.imshow(cv2.cvtColor(squares_image, cv2.COLOR_BGR2RGB))

        plt.subplot(132)
        plt.title("Geometrically valid squares\nvalid_squares_image")
        plt.imshow(valid_squares_image, cmap="gray")

        plt.subplot(133)
        plt.title("All detected quadrilaterals\nall_contours_image")
        plt.imshow(all_contours_image, cmap="gray")
        plt.show()
        #################################################
        # Apply dilation to the valid_squares_image
        kernel = np.ones((7, 7), np.uint8)
        dilated_valid_squares_image = cv2.dilate(valid_squares_image, kernel, iterations=1)  

        plt.figure(figsize=(12,8))
        plt.title("dilated_valid_squares_image")
        plt.imshow(dilated_valid_squares_image,cmap="gray")

        #################################################

        # Create a convex hull mask to fill the chessboard

        # Find the contours of the original mask
        contours, _ = cv2.findContours(dilated_valid_squares_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Create a new mask with the convex hull of the largest contour
            hull_mask = np.zeros_like(dilated_valid_squares_image)
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)
            cv2.drawContours(hull_mask, [hull], 0, 255, -1)
            
            # Display the convex hull mask
            plt.figure(figsize=(12, 8))
            plt.title("Convex Hull Mask")
            plt.imshow(hull_mask, cmap="gray")
            plt.show()
            
            # Apply the convex hull mask to get the final isolated chessboard
            filled_chessboard = cv2.bitwise_and(dilated_valid_squares_image, dilated_valid_squares_image, mask=hull_mask)
            
            # Display the result
            plt.figure(figsize=(12, 8))
            plt.title("Filled Chessboard (Convex Hull)")
            plt.imshow(filled_chessboard, cmap="gray")
            plt.show()
            
            # Update the chessboard mask and isolated image
            chessboard_mask = hull_mask
            dilated_valid_squares_image = filled_chessboard
        else:
            print("No contours found in the chessboard mask")


            ################################################################


        # Create a visualization image
        visual_img = cv2.cvtColor(chessboard_mask.copy(), cv2.COLOR_GRAY2BGR)

        # Find the contour of the chessboard (should be the largest contour)
        contours, _ = cv2.findContours(chessboard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No contours found!")
        else:
            # Get the largest contour
            main_contour = max(contours, key=cv2.contourArea)
            
            # Use the Douglas-Peucker algorithm to approximate the contour
            # This parameter controls how detailed the approximation is
            # Try different values to get exactly 4 points
            
            # Start with a higher epsilon and gradually decrease if needed
            for epsilon_factor in [0.02, 0.015, 0.01, 0.005]:
                epsilon = epsilon_factor * cv2.arcLength(main_contour, True)
                approx = cv2.approxPolyDP(main_contour, epsilon, True)
                
                print(f"With epsilon factor {epsilon_factor}, found {len(approx)} points")
                
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
                print("Not enough corners found, falling back to convex hull method")
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
            
            # Now we should have exactly 4 corners
            # Let's sort them in clockwise order for proper drawing
            
            # First, find the centroid of the corners
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
            
            # Display the results
            plt.figure(figsize=(12, 8))
            plt.title("Corner Detection (Ordered by Detection)")
            plt.imshow(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB))
            plt.show()
            
            # Print the corner coordinates in their original order
            for i, corner in enumerate(corners):
                print(f"Corner {i}: ({int(corner[0])}, {int(corner[1])})")
