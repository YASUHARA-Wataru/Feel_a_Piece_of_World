# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 19:33:49 2022

@author: YASUHARA Wataru
"""

import os
import glob
import cv2

frame_image_org_dir = r""
sub_dir = r"wave_shape_stereo"

frame_image_dir = frame_image_org_dir + sub_dir

Img_width = 160
Img_height = 120

frame_rate = 30
# 動画作成
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video  = cv2.VideoWriter(sub_dir+'.mov', fourcc, frame_rate, (int(Img_width),int(Img_height)))

for path_name in sorted(glob.glob(os.path.join(frame_image_dir+'\\','*jpg'))):
    img = cv2.imread(path_name)
    video.write(img)

video.release()