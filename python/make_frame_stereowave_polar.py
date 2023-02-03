# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:21:34 2022

@author: NavigateSafetyFiled
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size # 追加
from mpl_toolkits.axes_grid1.mpl_axes import Axes # 追加
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from module import LoadWav

def make_frame_seterewave_ploar(wave_data_file_name,
                          wave_data_folder,
                          img_output_folder,
                          fps,
                          au_display_sample_num,
                          output_width,
                          output_height,
                          file_num_zfill):
    
    au_data,au_data_time,a_sr = LoadWav.load_wav(wave_data_folder+wave_data_file_name)
    img_frame_times = np.arange(0,au_data_time[-1],1/fps)
    ch1_wav = au_data[0]
    ch2_wav = au_data[1]

    # サイズ指定のための処理 ↓↓ ここから ↓↓ 
    ax_w_px = output_width  # プロット領域の幅をピクセル単位で指定
    ax_h_px = output_height  # プロット領域の高さをピクセル単位で指定

    fig_dpi = 100
    ax_w_inch = ax_w_px / fig_dpi
    ax_h_inch = ax_h_px / fig_dpi
    ax_margin_inch = (0.5, 0.5, 0.5, 0.5)  # Left,Top,Right,Bottom [inch]
    
    fig_w_inch = ax_w_inch + ax_margin_inch[0] + ax_margin_inch[2] 
    fig_h_inch = ax_h_inch + ax_margin_inch[1] + ax_margin_inch[3]
        
    fig,ax4p = plt.subplots(1,2,dpi=fig_dpi, figsize=(fig_w_inch, fig_h_inch),subplot_kw={'projection': 'polar'})
    
    ax_p_w = [Size.Fixed(ax_margin_inch[0]),Size.Fixed(ax_w_inch)]
    ax_p_h = [Size.Fixed(ax_margin_inch[1]),Size.Fixed(ax_h_inch)]
    divider = Divider(fig, (0.0, 0.0, 1.0, 1.0), ax_p_w, ax_p_h, aspect=False)
    ax = Axes(fig, divider.get_position())
    ax.set_axes_locator(divider.new_locator(nx=1,ny=1))
    fig.add_axes(ax)
    ax.axis('off')
   
    plt.style.use('dark_background')

    for frame_index,img_frame_time in enumerate(img_frame_times):
        print("\r"+str(frame_index),end="")
        au_img_fit_ind = int(img_frame_time * a_sr)
        cut_index = np.arange(au_img_fit_ind - au_display_sample_num/2,au_img_fit_ind+au_display_sample_num/2,1)
        cut_index[cut_index < 0] = 0
        cut_index[cut_index >= au_data_time.shape[0]] = au_data_time.shape[0] - 1
        cut_index = cut_index.astype('int')
        
        x = np.linspace(0, 2 * np.pi, au_display_sample_num)

        enhance = 1.5    
        ax4p[0].plot(x,ch1_wav[cut_index]*enhance + 1,'c')
        ax4p[1].plot(x,ch2_wav[cut_index]*enhance + 1,'c')
        ax4p[0].set_ylim([1-enhance,2])
        ax4p[1].set_ylim([1-enhance,2])
        ax4p[0].axis('off')
        ax4p[1].axis('off')
        plt.savefig(img_output_folder+str(frame_index).zfill(file_num_zfill)+'.jpg',bbox_inches='tight',pad_inches=0)
        ax4p[0].cla()
        ax4p[1].cla()

            
def main():
    #wave_data_file_name = r"audio2image_test_music.wav"
    wave_data_file_name = r"test_stereo_diff.wav"
    wave_data_folder = r""
    #img_output_folder = r""
    img_output_folder = r""
    fps = 30
    au_display_sample_num = 1024
    #output_width = 1280
    #output_height = 720
    output_width = 640
    output_height = 361
    file_num_zfill = 10

    make_frame_seterewave_ploar(wave_data_file_name,
                          wave_data_folder,
                          img_output_folder,
                          fps,
                          au_display_sample_num,
                          output_width,
                          output_height,
                          file_num_zfill)
    
if __name__ == "__main__":   
    main()

