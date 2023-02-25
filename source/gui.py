import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from svm import Prediction
import os
from utilities import Utilities


global image_path

my_w = tk.Tk()
my_w.geometry("1000x600")  

my_w.title('Melanoma Detector')

my_font1=('times', 18, 'bold')

l1 = tk.Label(my_w,text='Upload Files & display',width=40,font=my_font1)  
l1.grid(row=1,column=1)

l3 = tk.Label(my_w,text="Result", width=40,font=my_font1)  
l3.grid(row=3,column=1) 

text = tk.Text(my_w, height = 5, width = 52)  
text.grid(row=3,column=2) 

browse_btn = tk.Button(my_w, text='browse', 
   width=20, command = lambda:upload_file())

browse_btn.grid(row=2,column=1)

predict_btn = tk.Button(my_w, text='predict Melanoma', 
   width=20,command = lambda:call_classifier())

predict_btn.grid(row=2,column=2)


def upload_file():
    f_types = [('Jpg Files', '*.jpg'),
    ('PNG Files','*.png')]   
    filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)
    
    col=1 
    row=3 
    for f in filename:
        img=Image.open(f) 
        img=img.resize((400,400)) 
        img=ImageTk.PhotoImage(img)
        e1 =tk.Label(my_w)
        e1.grid(row=row,column=col)
        e1.image = img 
        e1['image']=img  
        if(col==3): 
            row=row+1
            col=1    
        else:       
            col=col+1      

    global image_path 
    image_path = ''.join(filename)
    

def call_classifier(metadata_path='HAM10000_metadata.csv'):
    print("image_path=", image_path)

    img_number = Utilities.extract_img_number(image_path)

    dataset_path = os.path.dirname(image_path)
    
    result = Prediction.predict(dataset_path, metadata_path, img_number)
    
    if len(text.get("1.0", "end-1c")) == 0:                
        text.insert(tk.END, result)
    else:
        text.delete("1.0","end")
        text.insert(tk.END, result)


my_w.mainloop()  # Keep the window open