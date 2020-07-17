# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 18:41:37 2020

@author: Ahmed
"""





import tkinter as tk

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
from tkinter import filedialog as fd 
from PIL import Image, ImageTk
import imutils


face_cascade = cv2.CascadeClassifier('D:\\new_model\\FaceEmotion\\cascader\\haarcascade_frontalface_default.xml')


model = load_model('D:\\new_model\\FaceEmotion\\models\\my_Model.hdf5', compile=False)
Emotions = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]



def video():
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
            #emotion_probability = np.max(preds)
            label = Emotions[preds.argmax()]
    
     
        for (i, (emotion, prob)) in enumerate(zip(Emotions, preds)):
                    cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                  (204, 0, 204), 2)
    
        cv2.imshow('live', frameClone)
    
        if (cv2.waitKey(2000) & 0xFF == ord('q')):
            break
    
    cap.release()
    cv2.destroyAllWindows()





def saved_img():
    img_path= fd.askopenfilename() 
    orig_frame = cv2.imread(img_path) 
    frame = cv2.imread(img_path,0)
    faces = face_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = frame[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = model.predict(roi)[0]
        label = Emotions[preds.argmax()]
        cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (51, 51, 255), 2)
        cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH),(204, 0, 204), 2)
        

    cv2.imshow('test_face', orig_frame)
    if (cv2.waitKey(2000) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
    
    
    
    
    
    

    
root = tk.Tk()

root.title("FaceEmotion")
root.geometry("500x400")
root.configure(bg='#008080')

    
button1=tk.Button(root,text="Live",width=20,height=3,bg='#008080',
                  bd=5,font = "Helvetica 9 bold ",
                  command=lambda:video())
button2=tk.Button(root,text="Image",width=20,height=3,bg='#008080',
                  bd=5,font = "Helvetica 9 bold ",
                  command=lambda:saved_img())

#button3=tk.Button(root,text="Image",width=20,height=3,bg='#008080',
                 # bd=5,font = "Helvetica 9 bold ",
                #  command=lambda:openNewWindow())

leb=tk.Label(root,text="FaceEmotion",font = "Helvetica 16 bold "
                   ,fg = "#0C090A",bg='#008080')


img  = Image.open("D:\\new_model\\FaceEmotion\\img\\aast.png") 
photo=ImageTk.PhotoImage(img)
lab=tk.Label(image=photo,width=108,height=108,bg='#008080').place(x=15,y=15)








img1  = Image.open("D:\\new_model\\FaceEmotion\\img\\g.png") 
photo1=ImageTk.PhotoImage(img1)
lab1=tk.Label(image=photo1,width=60,height=60).place(x=40,y=150)

img2  = Image.open("D:\\new_model\\FaceEmotion\\img\\a.png") 
photo2=ImageTk.PhotoImage(img2)
lab2=tk.Label(image=photo2,width=60,height=60).place(x=100,y=150)

img3  = Image.open("D:\\new_model\\FaceEmotion\\img\\c.png") 
photo3=ImageTk.PhotoImage(img3)
lab3=tk.Label(image=photo3,width=60,height=60).place(x=160,y=150)

img4  = Image.open("D:\\new_model\\FaceEmotion\\img\\d.png") 
photo4=ImageTk.PhotoImage(img4)
lab4=tk.Label(image=photo4,width=60,height=60).place(x=220,y=150)

img5  = Image.open("D:\\new_model\\FaceEmotion\\img\\f.png") 
photo5=ImageTk.PhotoImage(img5)
lab5=tk.Label(image=photo5,width=60,height=60).place(x=280,y=150)

img6  = Image.open("D:\\new_model\\FaceEmotion\\img\\b.png") 
photo6=ImageTk.PhotoImage(img6)
lab6=tk.Label(image=photo6,width=60,height=60).place(x=340,y=150)

img7  = Image.open("D:\\new_model\\FaceEmotion\\img\\e.png") 
photo7=ImageTk.PhotoImage(img7)
lab7=tk.Label(image=photo7,width=60,height=60).place(x=400,y=150)




img8  = Image.open("D:\\new_model\\FaceEmotion\\img\\ccit.png") 
photo8=ImageTk.PhotoImage(img8)
lab8=tk.Label(image=photo8,width=108,height=108).place(x=380,y=15)



leb_a=tk.Label(root,text="angry",font = "Helvetica 10 bold "
                   ,fg = "#0C090A",bg='#008080')

leb_b=tk.Label(root,text="disgust",font = "Helvetica 10 bold "
                   ,fg = "#0C090A",bg='#008080')

leb_c=tk.Label(root,text="scared",font = "Helvetica 10 bold "
                   ,fg = "#0C090A",bg='#008080')

leb_d=tk.Label(root,text="happy",font = "Helvetica 10 bold "
                   ,fg = "#0C090A",bg='#008080')

leb_e=tk.Label(root,text="sad",font = "Helvetica 10 bold "
                   ,fg = "#0C090A",bg='#008080')

leb_f=tk.Label(root,text="surprised",font = "Helvetica 10 bold "
                   ,fg = "#0C090A",bg='#008080')

leb_g=tk.Label(root,text="neutral",font = "Helvetica 10 bold "
                   ,fg = "#0C090A",bg='#008080')






button1.pack()
button2.pack()

leb.pack()

leb_a.pack()

button1.place(x=70,y=330)
button2.place(x=285,y=330)


leb.place(x=190,y=20)
leb_a.place(x=47,y=220)
leb_b.place(x=110,y=220)
leb_c.place(x=170,y=220)
leb_d.place(x=230,y=220)
leb_e.place(x=295,y=220)
leb_f.place(x=340,y=220)
leb_g.place(x=415,y=220)


#tex.insert()
root.mainloop()

