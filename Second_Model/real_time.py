# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 13:31:14 2020

@author: Ahmed
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import imutils


face_cascade = cv2.CascadeClassifier('D:\\new_model\\FaceEmotion\\cascader\\haarcascade_frontalface_default.xml')


model = load_model('D:\\new_model\\FaceEmotion\\models\\my_Model.hdf5', compile=False)
Emotions = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]




cap = cv2.VideoCapture(0)
while True:
    frame = cap.read()[1]
    frame = imutils.resize(frame,width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

    frameClone = frame.copy()
    preds=[]
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                  
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = model.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = Emotions[preds.argmax()]

 
    for (i, (emotion, prob)) in enumerate(zip(Emotions, preds)):
                cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (204, 0, 204), 2)

    cv2.imshow('live', frameClone)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
