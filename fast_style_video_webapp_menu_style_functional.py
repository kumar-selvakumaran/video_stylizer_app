# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 14:34:19 2021

@author: Kumar
"""

import streamlit as st
# from io import StringIO  
# import pandas as pd
from PIL import Image
# import functools
import os
# import matplotlib
# from matplotlib import gridspec
# import matplotlib.pylab as plt
# import numpy as np
# import tensorflow as tf
# import tensorflow_hub as hub
# import cv2
# from video2frame import FrameCapture
# import glob
# import time
# from keras.preprocessing.image import save_img as save_frame
# from pathlib import Path
# import time 
from complete_working_pipeline import run_stylization_pipeline
# import random

st.title("VIDEO STYLIZATION using fast style transfer")

st.write("""
          Welcome to your own video neural-styling studio where you witness the artistic prowess of neural networks. 
         """)
   
st.sidebar.write("""
                 Patience is the key to great artworks , so naturally longer videos take more time to style ;) 
                 """)
            

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img 

def main():
    status = False #initial assignment
    menu = ["STYLE IMAGE","CONTENT VIDEO","STYLIZED VIDEO"]
    choice = st.sidebar.selectbox("Menu",menu)
    image_file = None
    video_file = None
    
    if choice == "STYLE IMAGE":
        st.subheader("STYLE IMAGE")
        image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
        if image_file is not None:
            # to See Details
			# st.write(type(image_file))
			# st.write(dir(image_file))
            file_details_image = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
# 			st.write(file_details_image)
                    
            img = load_image(image_file)
            style_image_path = "./style_images/style.jpg"
            if os.path.isfile(style_image_path):   # deleting previously existing style image if present
               os.remove(style_image_path)
            img.save(style_image_path)  #saving as style image
            st.image(img,width=400)     #viewing the image
    
    elif choice == "CONTENT VIDEO":
        st.subheader("CONTENT VIDEO")
        video_file = st.file_uploader("Upload the video",type=['mp4','avi'])
        if video_file is not None: 
            file_details_video = {"Filename":video_file.name,"FileType":video_file.type,"FileSize":video_file.size}
            st.write(file_details_video)
           
            """downloading the uploaded video as sample_video.mp4"""
            bytes_data = video_file.read()    # 'video_file' is an object of UploadedFile class which is a subclass of BytesIO , it is converted to bytes data in this line
            FILE_OUTPUT = 'sample_video.mp4'  #output path
            if os.path.isfile(FILE_OUTPUT):
                os.remove(FILE_OUTPUT)
            with open(FILE_OUTPUT, "wb") as out_file:  # open for [w]riting as [b]inary  (the bytes data is written as binary using 'wb' mode into a mp4 file)
                out_file.write(bytes_data) 
            st.video(bytes_data)    
                     
        if st.button('Stylize'):
            st.write('styling uploaded video to mosaic style')
            run_stylization_pipeline()
        else:
            st.write('press above button to style')
            
    elif choice == "STYLIZED VIDEO":
        st.subheader("STYLIZED VIDEO")
        stylized_video_path = './result_videos//HustlersFSS-style-content_frames.mp4'
        video_file = open(stylized_video_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
   

if __name__ == '__main__':
	main()



# """playing a video file"""
# video_file = open('./result_videos//HustlersFSS-retro-content_frames.mp4', 'rb')
# video_bytes = video_file.read()
# st.video(video_bytes)
