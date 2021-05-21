# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 18:03:16 2021

@author: Kumar

"""

import functools

import os

# import streamlit as st
# from matplotlib import gridspec
#import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from video2frame import FrameCapture
import glob
# import time
from keras.preprocessing.image import save_img as save_frame
# from pathlib import Path
# import time 

""" TENSORFLOW MODE REQUIRED FOR WORKFLOW (DONT CHANGE)"""

def set_tensorflow_mode():
    print("TF Version: ", tf.__version__)
    print("TF-Hub version: ", hub.__version__)
    print("Eager mode enabled: ", tf.executing_eagerly())
    print("GPU available: ", tf.test.is_gpu_available())

"""model import from tensorflow hub"""


def obtain_model():
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)  
    return hub_module

  
def get_paths():
    """ INPUT PATHS"""
    #for sample video
    # video_frames = './content_frames/*.png'
    
    #for arbitrary video
    video_frames = './content_frames/*.jpg'
    # style_image_path = './style_images/plum_flower.jpg'
    # style_image_path = './style_images/starry_night.jpg'
    # style_image_path = './style_images/Les_Demoiselles.jpg'
    # style_image_path = './style_images/mosaic.jpg'
    # style_image_path = './style_images/great_wave.jpg'
    # style_image_path = './style_images/tron.jpg'
    # style_image_path = './style_images/retro.jpg'
    """style from webapp"""
    style_image_path = './style_images/style.jpg'

    
    
    """OUTPUT PATHS"""
    result_frames_path = './result_frames/'
    result_videos_path = './result_videos/'
    
    """MAKING DIRECTORIES IF THEY DONT EXIST"""
    if not os.path.exists(result_frames_path):
        os.mkdir(result_frames_path)
    
    if not os.path.exists(result_videos_path):
        os.mkdir(result_videos_path)
    
    return (video_frames, 
            style_image_path, 
            result_frames_path, 
            result_videos_path)



"""---------------   helper functions   --------------"""

"""convert avi to mp4"""
def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i {input} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {output}.mp4".format(input = avi_file_path, output = output_name))
    return "convered to mp4 successfully"


def delete_result_frames():
    result_frames_path_del = './result_frames/*.jpg'
    files = glob.glob(result_frames_path_del)
    for f in files:
        os.remove(f)
        
def delete_result_videos():
    result_videos_path_del = './result_videos/*'
    files = glob.glob(result_videos_path_del)
    for f in files:
        os.remove(f)

def delete_content_frames():
    content_frames_path_del = './content_frames/*.jpg'
    files = glob.glob(content_frames_path_del)
    for f in files:
        os.remove(f)        
        
def video_capture(video_file_name):
    video = cv2.VideoCapture(video_file_name)
    if (video.isOpened() == False): 
        print("Error reading video file")
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('sample_video.avi', 
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, size)
    while(True):
        ret, frame = video.read()
        if ret == True: 
            result.write(frame)
        else:
            break
    video.release()
    result.release()
    cv2.destroyAllWindows()
    
    # """avi to mp4"""
    # avi_file_path = '{}/{}.avi'.format(result_videos_path, full_video_name)
    # mp4_file_path = '{}/{}'.format(result_videos_path, full_video_name)
    # convert_avi_to_mp4(avi_file_path, mp4_file_path)


"""--------------------------------------------------------"""        

""" CONTENT VIDEO TO FRAMES and OBTAINING THE CONTENT VIDEO'S FPS"""
def get_frames_from_content_video_path():
    content_init_path = "sample_video.mp4"
    delete_content_frames()
    fps = FrameCapture(path = content_init_path)
    return fps
    

""" TO SORT THE FILES IN ORDER OF FRAMES """
def file_sort_key(word):
    key = int((word.split('e')[-1]).split('.')[0])
    return key

"""getting frames location list"""
def get_content_frames_locations(video_frames , full_video = True):
    frame_list = glob.glob(video_frames)
    #sorting in order of frames
    frame_list.sort(key = file_sort_key)
    if full_video == True:    
        frame_num = len(frame_list)
    elif full_video == False:  
        frame_num = 100
    return (frame_list , frame_num)    
    

"""video naming"""
def get_video_name(style_image_path , video_frames):
    style_name = (style_image_path.split('/')[-1]).split('.')[0]
    video_name = (video_frames.split('/')[-2])
    full_video_name = 'HustlersFSS-' + style_name + '-' + video_name
    return (style_name,
            video_name,
            full_video_name)

""" Returns a cropped square image."""
def crop_center(image):
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

""" custom image loader """
@functools.lru_cache(maxsize=None)
def load_image(image_path, image_size=(540, 540), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
  if img.max() > 1.0:
    img = img / 255.
  if len(img.shape) == 3:
    img = tf.stack([img, img, img], axis=-1)
  # img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img


"""style image preprocessing"""
def get_processed_style_image(style_image_path):
    style_img_size = (540, 540)  # Recommended to keep it at 256.
    style_image = load_image(style_image_path, style_img_size)
    #pooling reduces computation expense and creates a less sofistcated styling
    # style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
    return (style_image,
            style_img_size)


"""stylizing frames"""
def stylize_content_frames(frame_num,
                           frame_list,
                           hub_module,  #the model from tensorflow hub
                           style_image,
                           result_frames_path):
    
    #deletes styled frames from previous stylization
    delete_result_frames()    
    #iterating over the frames
    for i in range(frame_num):
        print("Stylizing frame %d"%(i))
        # Read the image
        input_frame = load_image(frame_list[i])
        # Stylization
        outputs = hub_module(input_frame, style_image)
        stylized_image = outputs[0]
        # Save result
        save_frame('{}/{}'.format(result_frames_path,frame_list[i].split('/')[-1].replace(chr(92),"_")), stylized_image[0].numpy())



"""converting video"""
def convert_stylized_frames_to_video(result_frames_path,
                                     result_videos_path,
                                     full_video_name,
                                     fps = None,
                                     save_video = True):
    
    #if frame_capture dosent run , fps = None , placeholder/default fps = 30 
    if fps == None:
        fps = 30
    
    #deleting videos from previous stylization
    delete_result_videos()
    
    if save_video:
        frame_list = glob.glob("{}/*.*".format(result_frames_path))
        frame_list.sort(key = file_sort_key)
        demo = cv2.imread(frame_list[0])
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Also (*'DVIX') or (*'X264')
        videoWriter = cv2.VideoWriter('{}/{}.avi'.format(result_videos_path, full_video_name), 
                                        fourcc, fps, (demo.shape[1],demo.shape[0]))
        for frame in frame_list:
            videoWriter.write(cv2.imread(frame))
        videoWriter.release()
        
        # converting avi to mp4
        avi_file_path = '{}/{}.avi'.format(result_videos_path, full_video_name)
        mp4_file_path = '{}/{}'.format(result_videos_path, full_video_name)
        convert_avi_to_mp4(avi_file_path, mp4_file_path)
        return mp4_file_path
    

"""complete stylization pipeline"""    
def run_stylization_pipeline(use_existing_video = False , save_video = True):
    fps = None      #declaring fps 
    set_tensorflow_mode()
    model = obtain_model()
    (content_frames_path , style_image_path , result_frames_path , result_videos_path) = get_paths()
    if use_existing_video == False:
        fps = get_frames_from_content_video_path()
    (frame_list , num_of_frames) = get_content_frames_locations(content_frames_path , full_video = True)
    (style_name , video_name , full_video_name) = get_video_name(style_image_path , content_frames_path)       
    (style_image , style_img_size) =  get_processed_style_image(style_image_path)
    stylize_content_frames(num_of_frames, frame_list, model , style_image , result_frames_path)
    stylized_video_path = convert_stylized_frames_to_video(result_frames_path , result_videos_path, full_video_name, fps , save_video) 
    print("\n#\n###\n#########\n\nstylization successfully completed\n\n#########\n###\n#")
    
    
