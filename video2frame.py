# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:51:05 2021

@author: Kumar
"""


import cv2
# import glob
import os
# import scipy.io as scio
# import numpy as np
# import random
# import time
# from PIL import Image 

# import torch

result_frames_path = './content_frames/'

track_v_to_f = True

if not os.path.exists(result_frames_path):
    os.mkdir(result_frames_path)


  
# Function to extract frames 
def FrameCapture(path = None , video = None): 
      
        # Path to video file 
        if video == None:
            vidObj = cv2.VideoCapture(path) 
        if path == None:
            vidObj = video
        fps = round(vidObj.get(cv2.CAP_PROP_FPS))
        # Used as counter variable 
        count = 0
      
        # checks whether frames were extracted 
        success = 1
      
        while success: 
      
            # vidObj object calls read 
            # function extract frames 
            success, image = vidObj.read() 
      
            # Saves the frames with frame-count 
            try:
                cv2.imwrite(os.path.join(result_frames_path ,"frame%d.jpg" % count), image) 
                count += 1
                if track_v_to_f:
                    if count>0 and count%10 == 0:
                        print("converted {} frames ( v to f )".format(count))  
            
            except:
                print("finished | fps = {}".format(fps))
                return fps
        
  
# Driver Code 
#if __name__ == '__main__': 
  
# Calling the function 
# FrameCapture("sample_video.mp4") 
# for i in range(3):
#     print("complete")