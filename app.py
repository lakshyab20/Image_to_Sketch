# Importing required libraries

import cv2
import streamlit as st
import numpy as np
from PIL import Image


# Function for converting Color image to Pencil Sketch and other cartoon filters

def img_to_cartoon_filter(img,cartoon):
    
    #converting colored image to grayscale using cv2 cvtColor function
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # If cartoon filter is pencil sketch then getting the values for tuning image from user 
    if cartoon == 'Pencil Sketch':
        value = st.sidebar.slider('Brightness(high value, high brightness)',0.0,300.0,250.0)
        kernel = st.sidebar.slider('Boldness(high value, more bold edges)',1,99,25,step=2)
        
        # making image blur to handle noise in image
        gray_blur = cv2.GaussianBlur(gray,(kernel,kernel),0)
        
        # Dividing the image gives us a ratio of change between each pixel of two images
        cartoon=cv2.divide(gray,gray_blur,scale=value)
        
    # Detail Enhancement filter gives us a cartoon effect by sharpening the image, smoothing the colors, and enhancing the edges.   
    if cartoon == 'Detail Enhancement':
        
        smooth=st.sidebar.slider('Smoothness',3,99,5,step=2)
        kernel=st.sidebar.slider('Sharpness',1,21,3,step=2)
        enhance_edge=st.sidebar.slider('Enhancing Edges',0.0,1.0,0.5)
        
        # Blurring image using medianBlur (we can also use Gaussian Blur)
        gray=cv2.medianBlur(gray,kernel)
        
        # detecting the edges of the image
        edges=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
        
        # making image sharper using detailEnhance
        color = cv2.detailEnhance(img, sigma_s=smooth, sigma_r=enhance_edge)
        
        # finally doing detail enhancement
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        
    if cartoon == 'Pencil Edges':
        
        # Blurring the image
        kernel = st.sidebar.slider('Sharpness', 1, 99, 25, step=2)
        
        # Using Laplacian_filter for detecting the edges
        laplacian_filter=st.sidebar.slider('Edge Detection Intensity',3,9,3,step=2)
        
        # Noise Reduction from image for clear edges 
        noise_reduction=st.sidebar.slider('Noise',10,255,150)
        
        gray = cv2.medianBlur(gray, kernel)
        edges = cv2.Laplacian(gray, -1, ksize=laplacian_filter)
        
        edges_inv = 255-edges
    
        dummy, cartoon = cv2.threshold(edges_inv, noise_reduction, 255, cv2.THRESH_BINARY)
        
    if cartoon == "Bilateral Filter":
        
        
       
        smooth = st.sidebar.slider('Smoothness', 3, 99, 5, step=2)
        kernel = st.sidebar.slider('Sharpness', 1, 21, 3, step =2)
        enhance_edges = st.sidebar.slider('Edges', 1, 100, 50)
       
        gray = cv2.medianBlur(gray, kernel) 
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 9, 9)
    
        color = cv2.bilateralFilter(img, smooth, enhance_edges, smooth) 
        cartoon = cv2.bitwise_and(color, color, mask=edges) 

    return cartoon


# Heading of Web app
st.write("""
          # Image to Sketch !
          """
          )
# Info about web app
st.write('App for converting you Image to Pencil Sketch and different cartoon filters')

# For uploading a image by user
file = st.sidebar.file_uploader("Please upload an image", type=["jpg", "png",'jpeg'])

if file is None:
    st.text("Error! Please upload an image and then try again")
else:
    image = Image.open(file)
    img = np.array(image)
    
    option = st.sidebar.selectbox(
    'Select Cartoon Filter',
    ('Pencil Sketch', 'Detail Enhancement', 'Pencil Edges', 'Bilateral Filter'))
    
    st.text("Original image")
    st.image(image, use_column_width=True)
    
    st.text("Image after applying cartoon filters")
    cartoon = img_to_cartoon_filter(img, option)
    
    st.image(cartoon, use_column_width=True)
    
    

    
st.write("""
          ## Made By Lakshya Bhardwaj
          """
          )