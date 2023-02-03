# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 19:33:49 2022

@author: NavigateSafetyField
"""

import os
import glob
import cv2

frame_image_org_dir = r""
#sub_dir = r"wave_shape_stereo"
#sub_dir = r"polar_wave_stereo"
#sub_dir = r"circle_stereo_smooth"
#sub_dir = r"circle_stereo_smooth2"
#sub_dir = r"circle_stereo_dragon"
#sub_dir = r"circle_stereo_dragon2"
#sub_dir = r"circle_stereo_random"
#sub_dir = r"circle_stereo_smooth_not_inter"
#sub_dir = r"test_circle_stereo_smooth"
#sub_dir = r"test_circle_stereo_smooth2"
#sub_dir = r"test_circle_stereo_dragon"
#sub_dir = r"test_circle_stereo_dragon2"
#sub_dir = r"test_circle_stereo_smooth_not_inter"
sub_dir = r"cosmic_dark_ambient_2"

frame_image_dir = frame_image_org_dir + sub_dir

#Img_width = 1280
#Img_height = 720

#Img_width = 640
#Img_height = 360

Img_width = 640
Img_height = 260


#Img_width = 320
#Img_height = 240

#Img_width = 160
#Img_height = 120

frame_rate = 30
# 動画作成
#fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video  = cv2.VideoWriter(sub_dir+'.mov', fourcc, frame_rate, (int(Img_width),int(Img_height)))

for path_name in sorted(glob.glob(os.path.join(frame_image_dir+'\\','*jpg'))):
    img = cv2.imread(path_name)
    video.write(img)

video.release()