import cv2
import numpy as np

# Capture video feed
cap = cv2.VideoCapture(0)

# Loop over frames
while True:
    # Read frame
    ret, frame = cap.read()
    
    # Preprocess image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Select region of interest
    mask = np.zeros_like(edges)
    height, width = mask.shape
    polygon = np.array([[(0, height), (width, height), (width, height//2), (0, height//2)]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough Transform
    lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=150)
    
    # Extract lane lines
    left_lane_lines = []
    right_lane_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            if slope < 0:
                left_lane_lines.append((slope, intercept))
            else:
                right_lane_lines.append((slope, intercept))
    left_lane_lines_avg = np.average(left_lane_lines, axis=0)
    right_lane_lines_avg = np.average(right_lane_lines, axis=0)
    
    # Draw lane lines
    line_image = np.zeros_like(frame)
    if len(left_lane_lines) > 0:
        left_slope, left_intercept = left_lane_lines_avg
        y1 = height
        y2 = int(height / 2)
        x1 = int((y1 - left_intercept) / left_slope)
        x2 = int((y2 - left_intercept) / left_slope)
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    if len(right_lane_lines) > 0:
        right_slope, right_intercept = right_lane_lines_avg
        y1 = height
        y2 = int(height / 2)
        x1 = int((y1 - right_intercept) / right_slope)
        x2 = int((y2 - right_intercept) / right_slope)
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    
    # Merge lane lines with original image
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    
    # Show the result
    cv2.imshow('Lane Detection', result)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
