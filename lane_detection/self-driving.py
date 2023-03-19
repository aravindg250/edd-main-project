import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)
# capture = cv.VideoCapture(0)
# Check if camera opened successfully



def canny(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = 5
    blur = cv.GaussianBlur(gray, (kernel, kernel), 0)
    canny = cv.Canny(blur, 50, 160)
    return canny

def regionOfInterest(img):
    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros_like(img)
    triangle = np.array([[(200, height), (800, 350), (1200, height)]], np.int32)
    cv.fillPoly(mask, triangle, 255)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

def houghLines(img):
    houghLines = cv.HoughLinesP(img, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 10)
    return houghLines

# def displayLines(img, lines):
#     line_image = np.zeros_like(img)
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line.reshape(4)
#             # cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
#             cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 10)
#     return img


def displayLinesAVG(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
            # cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
                cv.line(img, (x1, y1), (x2, y2), (255, 255, 0), 10)
    return img


def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    if lines is not None:  
        for line in lines:
            for x1, y1, x2, y2 in line:
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0:
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))
        left_fit_avg = np.average(left_fit, axis=0)
        right_fit_avg = np.average(right_fit, axis=0)

        left_line = make_points(img, left_fit_avg)
        right_line = make_points(img, right_fit_avg)
        average_lines = [left_line, right_line]
        return average_lines

def make_points(img, lineSI):
    slope, intercept = lineSI
    height = img.shape[0]
    y1 = int(height)
    y2 = int(y1*3.0/5)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return [[x1, y1, x2, y2]]

#*************************************************************************
#*************************************************************************
#*************************************************************************
if (capture.isOpened()== False): 
    print("Error opening video stream or file")
 
# Read until video is completed
while(capture.isOpened()):
  # Capture frame-by-frame
    ret, frame = capture.read()

    canny_output = canny(frame)
    masked_output = regionOfInterest(canny_output)
    lines = houghLines(masked_output)
    average_lines = average_slope_intercept(frame, lines)
    line_image = displayLinesAVG(frame, average_lines)
    frame = line_image


    if ret == True:
 
    # Display the resulting frame
        cv.imshow('Frame',line_image)
        
    # Press Q on keyboard to  exit
    if cv.waitKey() & 0xFF == ord('q'):
        break
 
  # Break the loop
    else: 
        break
 
# When everything done, release the video capture object
capture.release()
 
# Closes all the frames
cv.destroyAllWindows()