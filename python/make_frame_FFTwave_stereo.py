# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:21:34 2022

@author: NavigateSafetyFiled
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size # 追加
from mpl_toolkits.axes_grid1.mpl_axes import Axes # 追加
from scipy import signal
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from module import LoadWav

print(__file__)
def make_frame_FFTwave(wave_data_file_name,
                          wave_data_folder,
                          img_output_folder,
                          fps,
                          display_sec,
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
        
    fig = plt.figure( dpi=fig_dpi, figsize=(fig_w_inch, fig_h_inch))
    ax_p_w = [Size.Fixed(ax_margin_inch[0]),Size.Fixed(ax_w_inch)]
    ax_p_h = [Size.Fixed(ax_margin_inch[1]),Size.Fixed(ax_h_inch)]
    divider = Divider(fig, (0.0, 0.0, 1.0, 1.0), ax_p_w, ax_p_h, aspect=False)
    ax = Axes(fig, divider.get_position())
    ax.set_axes_locator(divider.new_locator(nx=1,ny=1))
    fig.add_axes(ax)
    plt.style.use('dark_background')

    for frame_index,img_frame_time in enumerate(img_frame_times):
        print("\r"+str(frame_index),end="")
        au_img_fit_first = int((img_frame_time - display_sec/2 )* a_sr)
        au_img_fit_end = int((img_frame_time +  display_sec/2 )* a_sr)
        cut_index = np.arange(au_img_fit_first,au_img_fit_end,1)
        cut_index[cut_index < 0] = 0
        cut_index[cut_index >= au_data_time.shape[0]] = au_data_time.shape[0] - 1
        cut_index = cut_index.astype('int')
        
        ch1_FFT_wav = ch1_wav[cut_index]
        ch2_FFT_wav = ch2_wav[cut_index]

        N = 1024
        over_lap = 50
        ch1_freqs, times, ch1_Sx = signal.spectrogram(ch1_FFT_wav, fs=a_sr, 
                                                  nperseg=N, noverlap=over_lap,
                                                  detrend=False, scaling='spectrum') # スペクトログラム変数

        ch2_freqs, times, ch2_Sx = signal.spectrogram(ch2_FFT_wav, fs=a_sr,
                                                  nperseg=N, noverlap=over_lap,
                                                  detrend=False, scaling='spectrum') # スペクトログラム変数

        Sx = np.concatenate([ch1_Sx[:int(N/4),:].T,ch2_Sx[:int(N/4),:].T],axis=1)
         
        ax.pcolor(Sx, cmap='jet')
        plt.axis('off')
        plt.savefig(img_output_folder+str(frame_index).zfill(file_num_zfill)+'.jpg',bbox_inches='tight',pad_inches=0)
        plt.cla()

            
def main():
    wave_data_file_name = r"audio2image_test_music.wav"
    wave_data_folder = r""
    img_output_folder = r""
    fps = 30
    display_sec = 1
    
    output_width = 1280
    output_height = 720
    #output_width = 320
    #output_height = 240
    file_num_zfill = 10

    make_frame_FFTwave(wave_data_file_name,
                       wave_data_folder,
                       img_output_folder,
                       fps,
                       display_sec,
                       output_width,
                       output_height,
                       file_num_zfill)
    
if __name__ == "__main__":   
    main()

