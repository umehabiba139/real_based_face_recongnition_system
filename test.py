from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import numpy as np
import os

video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data\haarcascade_frontalface_default.xml')

with open ('data/names.pkl','rb') as f:
      LABLES=pickle.load(f)

with open ('data/faces_data.pkl','rb') as f:
      fACES=pickle.load(f)

KNeighborsClassifier()

while True:
    ret,frame=video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img,(50,50))  
        cv2.rectangle(frame, (x,y),(x+w , y+h),(50,50,255),1)

    cv2.imshow("frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break 
video.release()
cv2.destroyAllWindows()


     



