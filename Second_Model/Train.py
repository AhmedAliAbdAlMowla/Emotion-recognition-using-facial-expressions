# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:49:32 2020

@author: Ahmed
"""



import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2
from network import train_network



image_size=(48,48)
batch_size = 32
num_epochs = 200
input_shape = (48, 48, 1)
validation_split = .2
num_classes = 7



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


train_network(xtrain,ytrain,xtest,ytest,num_classes,batch_size,num_epochs)





