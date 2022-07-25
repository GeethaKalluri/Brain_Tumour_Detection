import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import matplotlib
import cv2
matplotlib.use('Agg')

#load the trained model to classify the images
from keras.models import load_model
#http://localhost:8888/edit/Downloads/braintumor.h5
model = load_model(r'C:\Users\Nikhila\Downloads\braintumor.h5')
#dictionary to label all the CIFAR-10 dataset classes.
c= [
'giloma',
'meningoma',
'normal',
'pituitary',
]
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Brain Tumor Classification ')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)
def classify(file_path):
#global label_packed
#image = Image.open(file_path)
#image = image.resize((250,250))
#image = numpy.expand_dims(image, axis=0)
#image = numpy.array(image)
#image=image.reshape(-1,250,250,3)
IMG_SIZE=250
IMG_ARRAY=cv2.imread(file_path)
new_array=cv2.resize(IMG_ARRAY,(IMG_SIZE,IMG_SIZE))
image=new_array.reshape(-1,IMG_SIZE,IMG_SIZE,3)
prediction = model.predict([image])[0]
classes=prediction.argmax(axis=-1)
index=int(classes)
print(index)
sign = c[index]

print(sign)
label.configure(foreground='#011638', text=sign)
def show_classify_button(file_path):
classify_b=Button(top,text="Classify Image",
command=lambda: classify(file_path),padx=10,pady=5)
classify_b.configure(background='#364156', foreground='red',
font=('arial',10,'bold'))
classify_b.place(relx=0.79,rely=0.46)
def upload_image():
try:
file_path=filedialog.askopenfilename()
uploaded=Image.open(file_path)
uploaded.thumbnail(((top.winfo_width()/2.25),
(top.winfo_height()/2.25)))
im=ImageTk.PhotoImage(uploaded)
sign_image.configure(image=im)
sign_image.image=im
label.configure(text='')
show_classify_button(file_path)
except:
pass
upload=Button(top,text="Upload an image",command=upload_image,
padx=10,pady=5)
upload.configure(background='#364156', foreground='white',
font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Brain Tumor Classification",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
