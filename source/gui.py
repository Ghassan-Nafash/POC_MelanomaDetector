import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from svm import Prediction
import os
from utilities import Utilities


global image_path

frame = tk.Tk()
frame.geometry("1100x600")  

frame.title('Melanoma Detector')

my_font1=('times', 18, 'bold')

my_font2=('times', 16, 'bold')

frame.columnconfigure(0, weight=5)
frame.columnconfigure(1, weight=5)

l1 = tk.Label(frame,text='Melanoma Detector',width=40,font=my_font1)  
l1.place(x=250, y=20)

browse_btn = tk.Button(frame, text='browse', width=20, command = lambda:upload_file(), font=my_font2)
browse_btn.place(x=25, y=100)

predict_btn = tk.Button(frame, text='predict Melanoma', width=20,command = lambda:call_classifier(), font=my_font2)
predict_btn.place(x=25, y=150)

Prediction_label = tk.Label(frame,text="Prediction", width=10, font=my_font1)  
Prediction_label.place(x=5, y=200)
Prediction_text = tk.Text(frame, height = 2, width = 15, font=my_font2)  
Prediction_text.place(x=180, y=200)

Correctness_Label = tk.Label(frame,text="Correctness", width=10, font=my_font1)  
Correctness_Label.place(x=5, y=250)
Correctness_text = tk.Text(frame, height = 2, width = 10, font=my_font2)  
Correctness_text.place(x=180, y=260)


def upload_file():
    f_types = [('Jpg Files', '*.jpg'),
    ('PNG Files','*.png')]   
    filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)

    for file in filename:
        img=Image.open(file) 

        img=img.resize((400,400)) 

        img=ImageTk.PhotoImage(img)

        e1 =tk.Label(frame)

        e1.place(x=400, y=100)

        e1.image = img 
        
        e1['image']=img  

    global image_path 
    image_path = ''.join(filename)
    

def call_classifier(metadata_path='HAM10000_metadata.csv'):

    img_number = Utilities.extract_img_number(image_path)

    dataset_path = os.path.dirname(image_path)
    
    prediction, correctness = Prediction.predict(dataset_path, metadata_path, img_number)

    # encode prediction result 1: Melanoma, 0: Not Melanoma
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

# Keep the window open
frame.mainloop()  