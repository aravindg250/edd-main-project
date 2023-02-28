import cv2 as cv
import numpy as np

# Libraries for Raspberry Pi and sensors
# import serial,time
# import RPi.GPIO as GPIO
# import tfluna_config as tfluna

# Example setup
# leftBuzzer = 22
# rightBuzzer = 13
# GPIO.setup(leftBuzzer, GPIO.OUT)
# GPIO.setup(rightBuzzer, GPIO.OUT)

cascade_model = cv.CascadeClassifier('cars.xml')

# Detects cars using Haar Cascade Classification returning positions of cars as rectangles (scale factor from 1 to 2 determining the detailedness), minNeighbors determining the confidence of the classifications, minSize determining the size of the bounding box (the car)
def detect_cars(image):
    return cascade_model.detectMultiScale3(image, scaleFactor=1.05, minNeighbors=20, minSize=(150,150), outputRejectLevels=1)

# Grays and blurs a BGR frame
def cvt_frame(frame, kernel_size):

    # Graying frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Blurring frame (kernel_size is degree of blurredness)
    gray = cv.GaussianBlur(src=gray, ksize= (kernel_size, kernel_size), sigmaX=0)

    return gray

# Video Based
capture = cv.VideoCapture("videos/single_car_parkway1.mp4")

while True:
    _, frame = capture.read()
    processed_frame = cvt_frame(frame, 1)
     
    cars_loc = detect_cars(processed_frame)
      
    # Code for finding the position of the car with the greatest confidence interval (level weight returned from outputRejectValues)
    index = 0
    for confidence in cars_loc[2]:
        if confidence == max(cars_loc[2]):
            (x,y,w,h) = cars_loc[0][index]
            cv.rectangle(processed_frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
            cv.putText(processed_frame, f'Car at ({x},{y})', (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)
            
            # put some code for considering the distance from the car and notifying user if necessary
            # (distance, strength, temperature) = tfluna.read_tfluna_data()
            # if (distance <10):
            #     GPIO.output(leftBuzzer, GPIO.HIGH)
            #     print("watch you left!")
            # else:
            #     GPIO.output(leftBuzzer, GPIO.LOW)
            
            print("Theres a car")

            break
        else:
            
            index = index + 1

        

    # Code for finding all of the positions of the cars
    # for (x,y,w,h) in cars_loc:
    #     cv.rectangle(processed_frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
    #     cv.putText(processed_frame, f'Car at ({x},{y})', (x,y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    resized_frame = cv.resize(processed_frame, (int(processed_frame.shape[1])//4, int(processed_frame.shape[0])//4))

    cv.imshow('Video', resized_frame)

    if cv.waitKey(20) & 0xFF==ord('q'):
        break

capture.release()
cv.destroyAllWindows()

# Image Based

# img = cv.imread('images/Parkway1.jpg')
# processed_img = cvt_frame(img, 1)
# cars_loc = detect_cars(processed_img)

# Code for finding the position of the car with the greatest confidence interval
# index = 0
# for confidence in cars_loc[2]:
#     if confidence == max(cars_loc[2]):
#         (x,y,w,h) = cars_loc[0][index]
#         cv.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 2)
#         cv.putText(img, f'Car at ({x},{y})', (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)
#         break
#     else:
#         index = index + 1

# Code for finding all of the positions of the cars
# for (x,y,w,h) in cars_loc[0]:
#     cv.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 2)
#     cv.putText(img, f'Car at ({x},{y})', (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)

# cv.imshow('image', img)

# cv.waitKey(0)