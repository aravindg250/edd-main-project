import cv2 as cv
import numpy as np


cascade_model = cv.CascadeClassifier('cars.xml')

# Detects cars using Haar Cascade Classification returning positions of cars as rectangles (scale factor from 1 to 2 determining the detailedness)
def detect_cars(image):
    return cascade_model.detectMultiScale(image, scaleFactor=1.05, minNeighbors=8, minSize=(25,25))

# Grays and blurs a BGR frame
def cvt_frame(frame, kernel_size):

    # Graying frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Blurring frame (kernel_size is degree of blurredness)
    gray = cv.GaussianBlur(src=gray, ksize= (kernel_size, kernel_size), sigmaX=0)

    return gray

# Video Based
capture = cv.VideoCapture("videos/single_car2.mp4")

while True:
    _, frame = capture.read()
    processed_frame = cvt_frame(frame, 1)
     
    cars_loc = detect_cars(processed_frame)

    for (x,y,w,h) in cars_loc:
        cv.rectangle(processed_frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
        cv.putText(processed_frame, f'Car at ({x},{y})', (x,y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    resized_frame = cv.resize(processed_frame, (int(processed_frame.shape[1]//4), int(processed_frame.shape[0]//4)))

    cv.imshow('Video', resized_frame)

    if cv.waitKey(20) & 0xFF==ord('q'):
        break

capture.release()
cv.destroyAllWindows()

# Image Based

# img = cv.imread('images/cars_on_highway.jpg')
# processed_img = cvt_frame(img, 1)
# cars_loc = detect_cars(processed_img)

# for (x,y,w,h) in cars_loc:
#     cv.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 2)
#     cv.putText(img, f'Car at ({x},{y})', (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)

# cv.imshow('image', img)

# cv.waitKey(0)