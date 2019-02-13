import cv2
import sys
import os
import numpy as np

# Displays a continuous video stream.
def webcam_show(mirror):
    cam = cv2.VideoCapture(0)

    while True:
        ret_val, img = cam.read()

        # Mirrors the image if our mirror variable is set to true.
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow("Webcam", img)

        if cv2.waitKey(1) == 27:
            break
    
    cv2.destroyAllWindows()
        
def face_detection(frame):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale (
        frame,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def eye_detection(frame):
    eye_cascade = cv2.CascadeClassifier('/home/zack/Desktop/PythonCV2/opencv/data/haarcascades/haarcascade_eye.xml')
    

if __name__ == "__main__":
    file_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(file_path)
    video_capture = cv2.VideoCapture(0)
    eye_cascade = cv2.CascadeClassifier('/home/zack/Desktop/PythonCV2/opencv/data/haarcascades/haarcascade_eye.xml')

    # This loop allows us to view the input from the camera in real time.
    while True:
        ret_val, frame = video_capture.read()

        # Converting the video to grayscale.
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(grey, 1.3, 5)
        # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = grey[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
           # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0, 255, 0), 2)

        cv2.imshow("Video", frame)
        # Detecting the faces.
        # faces = face_detection(frame)
        # eyes = eye_detection(frame)

        '''
        for (x, y, w, h) in faces:
            roi_grey = grey[y:y+h, x:x+w]
            eyes = cv2.CascadeClassifier("haarcascade_eye.xml")     
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle()
        '''
        
        # Drawing rectangles around the faces (greyscale & normal)
        for (x, y, w, h) in faces:
            cv2.rectangle(grey, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Displaying the greyscale image.
        cv2.imshow("Greyscale Image", grey)

        cv2.imshow("Normal Image", frame)
        
        if cv2.waitKey(10) == 27:
            break

    # Clean up.
    video_capture.release()
    cv2.destroyAllWindows()
