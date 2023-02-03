# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 19:33:49 2022

@author: NavigateSafetyField
"""

import os
import glob
import cv2

frame_image_org_dir = r""
"""
sub_dir1 = r"polar_wave_stereo"
sub_dir2 = r"wave_shape_stereo"
sub_dir3 = r"FFT_stereo"
sub_dir4 = r"cirlcle_stereo"

sub_dir1 = r"circle_stereo_dragon"
sub_dir2 = r"circle_stereo_smooth"
sub_dir3 = r"circle_stereo_dragon2"
sub_dir4 = r"circle_stereo_smooth2"
"""

sub_dir1 = r"test_circle_stereo_dragon"
sub_dir2 = r"test_circle_stereo_smooth"
sub_dir3 = r"test_circle_stereo_dragon2"
sub_dir4 = r"test_circle_stereo_smooth2"


#image_num = 3231
image_num = 989

Img_width = 1280
Img_height = 720

#Img_width = 320
#Img_height = 240

frame_rate = 30
# 動画作成
#fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('test_all.mov', fourcc, frame_rate,
                        (int(Img_width), int(Img_height)))


# print(cv2.getBuildInformation())

for num in range(image_num+1):
    img1 = cv2.imread(frame_image_org_dir+sub_dir1 +
                      '\\'+str(num).zfill(10)+'.jpg')
    img2 = cv2.imread(frame_image_org_dir+sub_dir2 +
                      '\\'+str(num).zfill(10)+'.jpg')
    img3 = cv2.imread(frame_image_org_dir+sub_dir3 +
                      '\\'+str(num).zfill(10)+'.jpg')
    img4 = cv2.imread(frame_image_org_dir+sub_dir4 +
                      '\\'+str(num).zfill(10)+'.jpg')
    upper_img = cv2.hconcat([img1, img2])
    lower_img = cv2.hconcat([img3, img4])
    img = cv2.vconcat([upper_img, lower_img])
    #img = cv2.resize(img,dsize=(Img_width,Img_height))
    video.write(img)

video.release()
