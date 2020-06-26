# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 23:46:32 2020

@author: Ahmed
"""



from keras.models import Sequential
from keras.layers import Dense, Dropout,  Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


def train_network(X_train,train_y,X_valid,valid_y,num_labels,batch_size,epochs):

    
    model = Sequential()
    
    model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
    model.add(Conv2D(8,kernel_size= (3, 3), activation='relu'))
   
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    
    
    
   
    model.add(Conv2D(16, kernel_size=(1,1), activation='relu'))
   
  
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
   
    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))
   
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    
    
    
    model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))
    
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
     
    
  
    model.add(Conv2D(128, (1, 1), activation='relu'))
   
    
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    
    model.add(Flatten())
    
    #fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(num_labels, activation='softmax'))
    
    model.summary()
    
    #Compliling the model
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])
    
    
    model.fit(X_train, train_y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_valid, valid_y),
              shuffle=True)
    
    model.save("my_Model.hdf5")