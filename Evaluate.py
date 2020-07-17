# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:36:24 2020

@author: Ahmed
"""



import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.models import load_model
import cv2


image_size=(48,48)



def load_Data():
	data =pd.read_csv('D:\\new_model\\FaceEmotion\\Data_set\\fer2013.csv')
	pixels = data['pixels'].tolist()
	width, height = 48, 48
	faces = []
	for pixel_sequence in tqdm(pixels):
		face = [int(pixel) for pixel in pixel_sequence.split(' ')]
		face = np.asarray(face).reshape(width, height)
		face = cv2.resize(face.astype('uint8'),image_size)
		faces.append(face.astype('float32'))
	faces = np.asarray(faces)
	faces = np.expand_dims(faces, -1)
	emotions = pd.get_dummies(data['emotion']).as_matrix()
	return faces, emotions

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x





faces, emotions = load_Data()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)


model = load_model('D:\\new_model\\FaceEmotion\\models\\my_Model.hdf5', compile=False)

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


l,acc=model.evaluate(xtest,ytest)


print("Accuracy= ",round(acc*100),"%")
