import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from svm import Prediction
import os
from utilities import Utilities
import numpy as np
import cv2 


frame = tk.Tk()
frame.geometry("900x600")  

frame.title('Melanoma Detector')

my_font1=('montserrat', 18)

my_font2=('montserrat', 16)

frame.columnconfigure(0, weight=5)
frame.columnconfigure(1, weight=5)

# define "global" variables
img_path = tk.StringVar()
dataset_path = tk.StringVar()
img_number = tk.IntVar()

l1 = tk.Label(frame,text='Melanoma Detector',width=40,font=my_font1)  
l1.place(x=150, y=20)

browse_btn = tk.Button(frame, text='Select image', width=15, command = lambda:upload_file(img_path, dataset_path, img_number), font=my_font2)
browse_btn.place(x=97, y=100)

predict_btn = tk.Button(frame, text='Predict melanoma', width=15,command = lambda:call_classifier(img_number, dataset_path), font=my_font2)
predict_btn.place(x=97, y=150)

Prediction_label = tk.Label(frame,text="Prediction:", width=10, font=my_font1)  
Prediction_label.place(x=5, y=200)
Prediction_text = tk.Text(frame, height = 2, width = 15, font=my_font2)  
Prediction_text.place(x=180, y=200)

Correctness_Label = tk.Label(frame,text="Correctness:", width=10, font=my_font1)  
Correctness_Label.place(x=5, y=250)
Correctness_text = tk.Text(frame, height = 2, width = 15, font=my_font2)  
Correctness_text.place(x=180, y=260)

# browse images in dataset
Browse_dataset_label = tk.Label(frame,text="Browse dataset:", width=30, font=my_font2)  
Browse_dataset_label.place(x=25, y=370)

l1 = tk.Label(frame,text='Melanoma Detector',width=40,font=my_font1)  
l1.place(x=150, y=20)

prev_btn = tk.Button(frame, text='Previous', width=10, command = lambda:next_img(-1, img_number, img_path, dataset_path), font=my_font2)
prev_btn.place(x=50, y=400)

next_btn = tk.Button(frame, text='Next', width=10,command = lambda:next_img(1, img_number, img_path, dataset_path), font=my_font2)
next_btn.place(x=200, y=400)


def next_img(step, img_number, img_path, dataset_path):
    new_img_number = img_number.get() + step
    list_of_images = [Utilities.extract_img_number(i) for i in Utilities.gen_file_names(dataset_path.get())]

    if new_img_number in list_of_images:
        img_number.set(new_img_number)
        img_path.set(dataset_path.get() + "\ISIC_00" + str(img_number.get()) + ".jpg")

        img=Image.open(img_path.get())

        img=img.resize((450,450)) 

        img=ImageTk.PhotoImage(img)

        e1 =tk.Label(frame)

        e1.place(x=400, y=100)

        e1.image = img 
        
        e1['image']=img

        Prediction_text.delete("1.0","end")
        Correctness_text.delete("1.0","end")



def upload_file(img_path, dataset_path, img_number):
    f_types = [('Jpg Files', '*.jpg'),
    ('PNG Files','*.png')]   
    filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)

    for file in filename:
        img=Image.open(file) 

        img=img.resize((450,450)) 

        img=ImageTk.PhotoImage(img)

        e1 =tk.Label(frame)

        e1.place(x=400, y=100)

        e1.image = img 
        
        e1['image']=img  

    img_path.set(''.join(filename))
    dataset_path.set(os.path.dirname(img_path.get()))
    img_number.set(Utilities.extract_img_number(img_path.get()))

    Prediction_text.delete("1.0","end")
    Correctness_text.delete("1.0","end")
    

def place_initial_image():
    placeholder_img = np.ones((450, 450))
    cv2.putText(placeholder_img, "No Image Loaded", (50, 80),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=0, thickness=3)

    img = Image.fromarray(np.uint8(placeholder_img))
    img=ImageTk.PhotoImage(img)

    e1 =tk.Label(frame)

    e1.place(x=400, y=100)

    e1.image = img 
    
    e1['image']=img
                    

def call_classifier(img_number, dataset_path, metadata_path='HAM10000_metadata.csv'):    
    prediction, correctness = Prediction.predict(dataset_path.get(), metadata_path, img_number.get())

    Prediction_text.delete("1.0","end")
    Correctness_text.delete("1.0","end")
    
    # encode prediction result 1: Melanoma, 0: Not Melanoma
    if prediction == None:
        prediction = "Not able to predict"
        correctness = ""
    if(prediction == 1):
        prediction = "Melanoma"                
    if(prediction == 0):
        prediction = "Not Melanoma"

    if len(Prediction_text.get("1.0", "end-1c") and Correctness_text.get("1.0", "end-1c")) == 0:        
        Prediction_text.insert(tk.END, prediction)
        Correctness_text.insert(tk.END, correctness)
    else:
        Prediction_text.delete("1.0","end")
        Correctness_text.delete("1.0","end")
        Prediction_text.insert(tk.END, prediction)
        Correctness_text.insert(tk.END, correctness)


place_initial_image()

# Keep the window open
frame.mainloop()  