import cv2
import numpy as np

# Capture video feed
cap = cv2.VideoCapture('test1.mp4')
live = cv2.VideoCapture(0)

prevoff = 0

# Loop over frames
while True:
    # Read frame
    ret, frame = cap.read()
    
    # Preprocess image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Select region of interest

    mask = np.zeros_like(edges)
    height, width = mask.shape
    polygon = np.array([[(200, height), (800, 350), (1200, height)]], dtype=np.int32)
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
        if abs(left_slope)>0.05:
            x1 = int((y1 - left_intercept) / left_slope)
            x2 = int((y2 - left_intercept) / left_slope)
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)

    # Bottom left coordinate of lane
        x_left = x1

    # Top left coordinate of lane
        x_tleft = x2

    if len(right_lane_lines) > 0:
        right_slope, right_intercept = right_lane_lines_avg
        y1 = height
        y2 = int(height / 2)
        if abs(right_slope)>0.05:
            x1 = int((y1 - right_intercept) / right_slope)
            x2 = int((y2 - right_intercept) / right_slope)
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)

    # Bottom right coordinate of lane
        x_right = x1
    
    # Top right coordinate of lane
        x_tright = x2
    
    # Car position and adding line at center of camera
    car_position = int(800)
    cv2.line(line_image, (car_position, height), (car_position, int(height/2)), (0,0,255), 10)

    # Conversion of pixels to meters (determined based on camera and position of camera)
    xmeters_pixels = 0.05 / 1000
    ymeters_pixels = 3.7 / 781

    if len(right_lane_lines)>0 and len(left_lane_lines)>0:
        # Adding the center line
        x_center = int((x_right + x_left)/2)
        cv2.line(line_image, (x_center, height), (x_center, int(height/2)), (255,0,0), 10)

        # Adding the line down the lane
        x_tcenter = int((x_tright+x_tleft)/2)
        cv2.line(line_image, (x_center, height), (x_tcenter, int(height/2)), (255,255,0), 10)

        # Calculating offset of car position from center of lane (assuming that the middle of the camera is the car position)
        offset = (np.abs(car_position) - np.abs(x_center)) * xmeters_pixels * 100

        #Ensuring the offset is within reasonable range
        if offset<3 and offset>-3:
            rounded_offset = np.round(offset, 1)
            cv2.putText(line_image, "offset: " + str(rounded_offset), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)
            
            prevoff = offset

        else:
    
            cv2.putText(line_image, "offset: " + str(prevoff), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)

    # Merge lane lines with original image
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    # Show the result
    cv2.imshow('Lane Detection', result)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()

cv2.destroyAllWindows()