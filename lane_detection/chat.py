import cv2
import numpy as np

# Define the region of interest (ROI) for lane detection
def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Perform Hough line detection on the image
def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)
    return lines

# Calculate the average slope and intercept of the detected lines
def average_slope_intercept(lines):
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))
    left_avg = np.average(left_lines, axis=0)
    right_avg = np.average(right_lines, axis=0)
    return left_avg, right_avg

# Calculate the lane offset
def calculate_offset(image, left_avg, right_avg):
    _, width, _ = image.shape
    left_slope, left_intercept = left_avg
    right_slope, right_intercept = right_avg
    y = height  # Assuming the bottom of the image is the nearest point
    left_x = (y - left_intercept) / left_slope
    right_x = (y - right_intercept) / right_slope
    center_offset = (left_x + right_x) / 2 - width / 2
    return center_offset

# Set up video capture
cap = cv2.VideoCapture(0)  # Use 0 for the first camera device

while True:
    _, frame = cap.read()
    height, width, _ = frame.shape

    # Define the region of interest (adjust these coordinates according to your needs)
    vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
    roi_frame = region_of_interest(frame, vertices)

    # Perform lane detection
    lines = detect_lines(roi_frame)
    if lines is not None:
        left_avg, right_avg = average_slope_intercept(lines)

        # Calculate offset
        offset = calculate_offset(frame, left_avg, right_avg)
        cv2.putText(frame, f"Offset: {offset:.2f} pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Lane Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()