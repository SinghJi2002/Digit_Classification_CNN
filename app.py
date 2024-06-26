import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image,ImageOps
import numpy as np
import keras

model = load_model("model.keras")
img_width=28
img_height=28
class_names={0: '0', 1: '1', 2: '2', 3: '3', 4: '4',5:'5',6:'6',7:'7',8:'8',9:'9'}

def prediction(img_array):
    cls=model.predict(img_array)
    return(cls)


def main():    
  st.title("Image Upload and Display")
  uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
  if uploaded_file is not None:
    
    # To read file as bytes:
    image = Image.open(uploaded_file).resize((img_width, img_height))
    image=ImageOps.grayscale(image)
    pic_arr = np.array(image)
      
    # Display the image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
      
    #Image Preprocessing
    img_tensor = pic_arr.reshape(1,img_height,img_width,1)
    cls=prediction(img_tensor)
    label = class_names[np.argmax(cls)]
    st.success(label)
      
if __name__ == '__main__':
  main()

    