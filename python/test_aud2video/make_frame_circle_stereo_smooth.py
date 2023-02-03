# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:21:34 2022

@author: NavigateSafetyFiled
"""

import numpy as np
from scipy import signal
from scipy import stats
from PIL import Image
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from module import LoadWav

def make_frame_cir(wave_data_file_name,
                          wave_data_folder,
                          img_output_folder,
                          fps,
                          au_display_sample_num,
                          display_freq_max,
                          delay_time,
                          output_width,
                          output_height,
                          file_num_zfill):
    
    au_data,au_data_time,a_sr = LoadWav.load_wav(wave_data_folder+wave_data_file_name)
    img_frame_times = np.arange(0,au_data_time[-1]+1/a_sr,1/fps)

    ch1_wav = au_data[0]
    ch2_wav = au_data[1]
    
    ch1_hilbert_signal = signal.hilbert(ch1_wav)
    ch2_hilbert_signal = signal.hilbert(ch2_wav)
    
    ch1_power = []
    ch2_power = []
    ch1_Freq_peak = []
    ch2_Freq_peak = []
    #ch1_peak_phase = []
    #ch2_peak_phase = []
    ch1_var = []
    ch2_var = []    
    ch1_skew = []
    ch2_skew = []    
    ch1_kurt = []
    ch2_kurt = []
    
    ch1_demod_phase =[]
    ch2_demod_phase =[]

    for frame_index,img_frame_time in enumerate(img_frame_times):
        au_img_fit_ind = int(img_frame_time* a_sr)
        print("\r"+str(frame_index)+":"+str(au_img_fit_ind),end="")
        cut_index = np.arange(int(au_img_fit_ind)-int(au_display_sample_num),int(au_img_fit_ind),1)
        cut_index[cut_index < 0] = 0
        cut_index[cut_index >= au_data_time.shape[0]] = au_data_time.shape[0] - 1
        cut_index = cut_index.astype('int')
        
        ch1_FFT_wav = ch1_wav[cut_index]
        ch2_FFT_wav = ch2_wav[cut_index]

        ch1_m_data = np.abs(ch1_hilbert_signal[cut_index])
        ch2_m_data = np.abs(ch2_hilbert_signal[cut_index])
        
        ch1_var.append(np.var(ch1_m_data))
        ch2_var.append(np.var(ch2_m_data))

        if np.sum(ch1_m_data-np.mean(ch1_m_data)) < pow(10,-8):
            ch1_skew.append(np.nan)
            ch1_kurt.append(np.nan)
        else:
            ch1_skew.append(stats.skew(ch1_m_data))
            ch1_kurt.append(stats.kurtosis(ch1_m_data))
        if np.sum(ch2_m_data-np.mean(ch2_m_data)) < pow(10,-8):
            ch2_skew.append(np.nan)
            ch2_kurt.append(np.nan)
        else:
            ch2_skew.append(stats.skew(ch2_m_data))
            ch2_kurt.append(stats.kurtosis(ch2_m_data))        


        ch1_FFT_data = np.fft.fft(ch1_FFT_wav)[:int(ch1_FFT_wav.shape[0]/2)]
        ch2_FFT_data = np.fft.fft(ch2_FFT_wav)[:int(ch2_FFT_wav.shape[0]/2)]
        
        ch1_FFT_data[-1] = 0+0j
        ch2_FFT_data[-1] = 0+0j
        
        ch1_Max_freq_ind = np.argmax(np.abs(ch1_FFT_data))
        ch2_Max_freq_ind = np.argmax(np.abs(ch2_FFT_data))

        ch1_freq = a_sr/2/(ch1_FFT_wav.shape[0]/2)*(ch1_Max_freq_ind)
        ch2_freq = a_sr/2/(ch2_FFT_wav.shape[0]/2)*(ch2_Max_freq_ind)
        
        if ch1_freq > display_freq_max:
            ch1_freq = display_freq_max
        if ch2_freq > display_freq_max:
            ch2_freq = display_freq_max

        mod_t = np.linspace(0, au_display_sample_num*(1/a_sr),au_display_sample_num)
        frame_init_time = au_img_fit_ind/a_sr
        ch1_demod_sin_wav = np.sin(2*np.pi*ch1_freq*mod_t)
        ch1_demod_cos_wav = np.cos(2*np.pi*ch1_freq*mod_t)
        ch1_I = ch1_FFT_wav[:au_display_sample_num]*ch1_demod_sin_wav
        ch1_Q = ch1_FFT_wav[:au_display_sample_num]*ch1_demod_cos_wav
        
        fp = 15*np.power(10,3)
        fs = 21*np.power(10,3)
        gpass = 3       #通過域端最大損失[dB]
        gstop = 40      #阻止域端最小損失[dB]
        if ch1_freq == 0:
            ch1_demod_phase.append(0)
            ch1_power.append(0)
        else:
            ch1_I_filted = lowpass(ch1_I,a_sr,fp,fs, gpass, gstop)
            ch1_Q_filted = lowpass(ch1_Q,a_sr,fp,fs, gpass, gstop)
            ch1_demod_phase_temp = np.arctan2(ch1_I_filted,ch1_Q_filted)[np.argmax(np.power(ch1_I_filted,2)+np.power(ch1_Q_filted,2))]
            ch1_init_phase = (np.mod(frame_init_time,1/ch1_freq)*ch1_freq)*2*np.pi
            if ch1_init_phase > np.pi:
                ch1_init_phase = -2*np.pi + ch1_init_phase
            ch1_demod_phase.append(ch1_demod_phase_temp +ch1_init_phase)
            ch1_power.append(np.sqrt(np.max(np.power(ch1_I_filted,2)+np.power(ch1_Q_filted,2))))

        ch2_demod_sin_wav = np.sin(2*np.pi*ch2_freq*mod_t)
        ch2_demod_cos_wav = np.cos(2*np.pi*ch2_freq*mod_t)

        ch2_I = ch2_FFT_wav[:au_display_sample_num]*ch2_demod_sin_wav
        ch2_Q = ch2_FFT_wav[:au_display_sample_num]*ch2_demod_cos_wav
        
        if ch2_freq == 0:
            ch2_demod_phase.append(0)
            ch2_power.append(0)
        else:
            ch2_I_filted = lowpass(ch2_I,a_sr,fp,fs, gpass, gstop)
            ch2_Q_filted = lowpass(ch2_Q,a_sr,fp,fs, gpass, gstop)
            ch2_demod_phase_temp = np.arctan2(ch2_I_filted,ch2_Q_filted)[np.argmax(np.power(ch2_I_filted,2)+np.power(ch2_Q_filted,2))]
            ch2_init_phase = (np.mod(frame_init_time,1/ch2_freq)*ch2_freq)*2*np.pi
            if ch2_init_phase > np.pi:
                ch2_init_phase = -2*np.pi + ch2_init_phase
            ch2_demod_phase.append(ch2_demod_phase_temp +ch2_init_phase)
            ch2_power.append(np.sqrt(np.max(np.power(ch2_I_filted,2)+np.power(ch2_Q_filted,2))))

        ch1_Freq_peak.append(ch1_freq)
        ch2_Freq_peak.append(ch2_freq)

    print("")
    
    ch1_power = np.log10(min_max_Normalization(np.array(ch1_power))+1)/np.log10(2)
    ch2_power = np.log10(min_max_Normalization(np.array(ch2_power))+1)/np.log10(2)
    ch1_Freq_peak = (np.array(ch1_Freq_peak))/display_freq_max
    ch2_Freq_peak = (np.array(ch2_Freq_peak))/display_freq_max
    ch1_peak_phase = np.mod((np.array(ch1_demod_phase) + np.pi),(2*np.pi))/(2*np.pi)
    ch2_peak_phase = np.mod((np.array(ch2_demod_phase) + np.pi),(2*np.pi))/(2*np.pi)
    ch1_vars = min_max_Normalization(np.array(ch1_var))
    ch2_vars = min_max_Normalization(np.array(ch2_var))
    
    ch1_skew = np.nan_to_num(ch1_skew,nan=np.nanmean(ch1_skew))  
    ch2_skew = np.nan_to_num(ch2_skew,nan=np.nanmean(ch2_skew))  
    ch1_kurt = np.nan_to_num(ch1_kurt,nan=np.nanmean(ch1_kurt))  
    ch2_kurt = np.nan_to_num(ch2_kurt,nan=np.nanmean(ch2_kurt))  
 
    ch1_skews = min_max_Normalization(np.array(ch1_skew))
    ch2_skews = min_max_Normalization(np.array(ch2_skew))
    ch1_kurts = min_max_Normalization(np.array(ch1_kurt))
    ch2_kurts = min_max_Normalization(np.array(ch2_kurt))  
    print(np.sum(np.isnan(ch1_skew)))
    
    #max_size_per_width = 8
    #min_size_per_width = 100
    max_size_per_width = 12
    min_size_per_width = 120
    ch1_size_params = ch1_power*output_width/max_size_per_width+output_width/min_size_per_width
    ch2_size_params = ch2_power*output_width/max_size_per_width+output_width/min_size_per_width
    ch1_y_posi_params = np.log2(ch1_Freq_peak+1)*output_height
    ch2_y_posi_params = np.log2(ch2_Freq_peak+1)*output_height
    ch1_x_posi_params = ch1_peak_phase*output_width
    ch2_x_posi_params = ch2_peak_phase*output_width

    pre_flames_r = np.zeros((int(fps*delay_time),output_height,output_width))
    pre_flames_g = np.zeros((int(fps*delay_time),output_height,output_width))
    pre_flames_b = np.zeros((int(fps*delay_time),output_height,output_width))
    
    
    for index in range(ch1_size_params.shape[0]):
        #print("\r"+str(index),end="")
        if index ==0:
            pre_index = 0
        else:
            pre_index = index - 1

        temp_ch1_diff_x1 = np.abs(ch1_x_posi_params[pre_index] - ch1_x_posi_params[index])
        temp_ch1_diff_x2 = output_width - np.abs(ch1_x_posi_params[pre_index] - ch1_x_posi_params[index])
        if temp_ch1_diff_x1 > temp_ch1_diff_x2:
            ch1_diff_x = temp_ch1_diff_x2
        else:
            ch1_diff_x = temp_ch1_diff_x1

        temp_ch2_diff_x1 = np.abs(ch2_x_posi_params[pre_index] - ch2_x_posi_params[index])
        temp_ch2_diff_x2 = output_width - np.abs(ch2_x_posi_params[pre_index] - ch2_x_posi_params[index])
        if temp_ch2_diff_x1 > temp_ch2_diff_x2:
            ch2_diff_x = temp_ch2_diff_x2
        else:
            ch2_diff_x = temp_ch2_diff_x1

        ch1_diff_y = ch1_y_posi_params[pre_index] - ch1_y_posi_params[index]
        ch2_diff_y = ch2_y_posi_params[pre_index] - ch2_y_posi_params[index]
        ch1_m_distance = np.sqrt(np.power(ch1_diff_x,2)+np.power(ch1_diff_y,2))
        ch2_m_distance = np.sqrt(np.power(ch2_diff_x,2)+np.power(ch2_diff_y,2))

        interpre_multi = 1/2
        ch1_interpre_num = int(ch1_m_distance*interpre_multi)
        ch2_interpre_num = int(ch2_m_distance*interpre_multi)
        print("\r"+str(index)+":"+str(ch1_interpre_num+ch2_interpre_num),end="")

        if ch1_interpre_num == 0:
            ch1_interpre_num = 1
        if ch2_interpre_num == 0:
            ch2_interpre_num = 1

        ch1_x_index = np.tile(np.tile(np.arange(0,output_width),(output_height,1)),(ch1_interpre_num,1,1)).transpose(1,2,0)
        if temp_ch1_diff_x1 > temp_ch1_diff_x2:
            ch1_x_index = np.concatenate([ch1_x_index[:,int(output_width/2):,:],ch1_x_index[:,:int(output_width/2),:]],axis=1)
        ch1_y_index = np.tile(np.tile(np.arange(0,output_height),(output_width,1)).T,(ch1_interpre_num,1,1)).transpose(1,2,0)

        ch2_x_index = np.tile(np.tile(np.arange(0,output_width),(output_height,1)),(ch2_interpre_num,1,1)).transpose(1,2,0)
        if temp_ch2_diff_x1 > temp_ch2_diff_x2:
            ch2_x_index = np.concatenate([ch2_x_index[:,int(output_width/2):,:],ch2_x_index[:,:int(output_width/2),:]],axis=1)
        ch2_y_index = np.tile(np.tile(np.arange(0,output_height),(output_width,1)).T,(ch2_interpre_num,1,1)).transpose(1,2,0)
        
        ch1_size_param = np.linspace(ch1_size_params[pre_index], ch1_size_params[index],ch1_interpre_num)
        ch2_size_param = np.linspace(ch2_size_params[pre_index],ch2_size_params[index],ch2_interpre_num)
        ch1_y_posi_param = np.linspace(ch1_y_posi_params[pre_index],ch1_y_posi_params[index],ch1_interpre_num)
        ch2_y_posi_param = np.linspace(ch2_y_posi_params[pre_index],ch2_y_posi_params[index],ch2_interpre_num)

        if temp_ch1_diff_x1 > temp_ch1_diff_x2:
            begin_point = revers_point(ch1_x_posi_params[pre_index],output_width)
            end_point = revers_point(ch1_x_posi_params[index],output_width)
            ch1_x_posi_param = np.linspace(begin_point,end_point,ch1_interpre_num)
        else:
            ch1_x_posi_param = np.linspace(ch1_x_posi_params[pre_index],ch1_x_posi_params[index],ch1_interpre_num)

        if temp_ch2_diff_x1 > temp_ch2_diff_x2:
            begin_point = revers_point(ch2_x_posi_params[pre_index],output_width)
            end_point = revers_point(ch2_x_posi_params[index],output_width)
            ch2_x_posi_param = np.linspace(begin_point,end_point,ch2_interpre_num)
        else:
            ch2_x_posi_param = np.linspace(ch2_x_posi_params[pre_index],ch2_x_posi_params[index],ch2_interpre_num)

        ch1_var = np.linspace(ch1_vars[pre_index],ch1_vars[index],ch1_interpre_num)
        ch2_var = np.linspace(ch2_vars[pre_index],ch2_vars[index],ch2_interpre_num)
        ch1_skew = np.linspace(ch1_skews[pre_index],ch1_skews[index],ch1_interpre_num)
        ch2_skew = np.linspace(ch2_skews[pre_index],ch2_skews[index],ch2_interpre_num)
        ch1_kurt = np.linspace(ch1_kurts[pre_index],ch1_kurts[index],ch1_interpre_num)
        ch2_kurt = np.linspace(ch2_kurts[pre_index],ch2_kurts[index],ch2_interpre_num)

        ch1_r_index = np.power(ch1_x_index - np.tile(ch1_x_posi_param.reshape(1,1,ch1_interpre_num),(output_height,output_width,1)),2) + np.power(ch1_y_index - np.tile(ch1_y_posi_param.reshape(1,1,ch1_interpre_num),(output_height,output_width,1)),2)
        ch2_r_index = np.power(ch2_x_index - np.tile(ch2_x_posi_param.reshape(1,1,ch2_interpre_num),(output_height,output_width,1)),2) + np.power(ch2_y_index - np.tile(ch2_y_posi_param.reshape(1,1,ch2_interpre_num),(output_height,output_width,1)),2)
        
        ch1_col_chara_max = np.max([ch1_var,ch1_skew,ch1_kurt],axis=0)
        ch2_col_chara_max = np.max([ch2_var,ch2_skew,ch2_kurt],axis=0)
        ch1_col_chara_min = np.min([ch1_var,ch1_skew,ch1_kurt],axis=0)
        ch2_col_chara_min = np.min([ch2_var,ch2_skew,ch2_kurt],axis=0)
        
        ch1_multi_r = np.zeros_like(ch1_var)
        ch1_multi_g = np.zeros_like(ch1_kurt)
        ch1_multi_b = np.zeros_like(ch1_skew)
        excep_index = ch1_col_chara_max != ch1_col_chara_min
        ch1_multi_r[~excep_index] = 1
        ch1_multi_g[~excep_index] = 1
        ch1_multi_b[~excep_index] = 1
        ch1_multi_r[excep_index] = (ch1_var[excep_index] - ch1_col_chara_min[excep_index])/(ch1_col_chara_max[excep_index] - ch1_col_chara_min[excep_index]) /2 + 0.5
        ch1_multi_g[excep_index] = (ch1_kurt[excep_index] - ch1_col_chara_min[excep_index])/(ch1_col_chara_max[excep_index] - ch1_col_chara_min[excep_index]) /2 + 0.5
        ch1_multi_b[excep_index] = (ch1_skew[excep_index] - ch1_col_chara_min[excep_index])/(ch1_col_chara_max[excep_index] - ch1_col_chara_min[excep_index]) /2 + 0.5

        ch2_multi_r = np.zeros_like(ch2_var)
        ch2_multi_g = np.zeros_like(ch2_kurt)
        ch2_multi_b = np.zeros_like(ch2_skew)
        excep_index = ch2_col_chara_max != ch2_col_chara_min
        ch2_multi_r[~excep_index] = 1
        ch2_multi_g[~excep_index] = 1
        ch2_multi_b[~excep_index] = 1
        ch2_multi_r[excep_index] = (ch2_var[excep_index] - ch2_col_chara_min[excep_index])/(ch2_col_chara_max[excep_index] - ch2_col_chara_min[excep_index]) /2 + 0.5
        ch2_multi_g[excep_index] = (ch2_kurt[excep_index] - ch2_col_chara_min[excep_index])/(ch2_col_chara_max[excep_index] - ch2_col_chara_min[excep_index]) /2 + 0.5
        ch2_multi_b[excep_index] = (ch2_skew[excep_index] - ch2_col_chara_min[excep_index])/(ch2_col_chara_max[excep_index] - ch2_col_chara_min[excep_index]) /2 + 0.5


        ch1_data_r = np.max(np.exp(-ch1_r_index/np.tile(np.power(2*ch1_size_param*ch1_multi_r,2),(output_height,output_width,1))),axis=2)
        ch2_data_r = np.max(np.exp(-ch2_r_index/np.tile(np.power(2*ch2_size_param*ch2_multi_r,2),(output_height,output_width,1))),axis=2)
        ch1_data_g = np.max(np.exp(-ch1_r_index/np.tile(np.power(2*ch1_size_param*ch1_multi_g,2),(output_height,output_width,1))),axis=2)
        ch2_data_g = np.max(np.exp(-ch2_r_index/np.tile(np.power(2*ch2_size_param*ch2_multi_g,2),(output_height,output_width,1))),axis=2)
        ch1_data_b = np.max(np.exp(-ch1_r_index/np.tile(np.power(2*ch1_size_param*ch1_multi_b,2),(output_height,output_width,1))),axis=2)
        ch2_data_b = np.max(np.exp(-ch2_r_index/np.tile(np.power(2*ch2_size_param*ch2_multi_b,2),(output_height,output_width,1))),axis=2)

       
        img_data_r = np.zeros_like(ch1_data_r)
        img_data_g = np.zeros_like(ch1_data_g)
        img_data_b = np.zeros_like(ch1_data_b)

        img_data_r[ch1_data_r > ch2_data_r] = ch1_data_r[ch1_data_r > ch2_data_r]
        img_data_r[ch1_data_r <= ch2_data_r] = ch2_data_r[ch1_data_r <= ch2_data_r]
        img_data_g[ch1_data_g > ch2_data_g] = ch1_data_g[ch1_data_g > ch2_data_g]
        img_data_g[ch1_data_g <= ch2_data_g] = ch2_data_g[ch1_data_g <= ch2_data_g]
        img_data_b[ch1_data_b > ch2_data_b] = ch1_data_b[ch1_data_b > ch2_data_b]
        img_data_b[ch1_data_b <= ch2_data_b] = ch2_data_b[ch1_data_b <= ch2_data_b]

        pre_flames_r[1:,:,:] = pre_flames_r[0:-1,:,:]
        pre_flames_r[0,:,:] = img_data_r
        pre_flames_g[1:,:,:] = pre_flames_g[0:-1,:,:]
        pre_flames_g[0,:,:] = img_data_g
        pre_flames_b[1:,:,:] = pre_flames_b[0:-1,:,:]
        pre_flames_b[0,:,:] = img_data_b


        num_of_pre = pre_flames_r.shape[0]
        
        write_flames_r = np.zeros_like(pre_flames_r)
        write_flames_g = np.zeros_like(pre_flames_g)
        write_flames_b = np.zeros_like(pre_flames_b)
        
        for num,(pre_flame_r,pre_flame_g,pre_flame_b) in enumerate(zip(pre_flames_r,pre_flames_g,pre_flames_b)):
            multi = 1-num/num_of_pre
            write_flames_r[num] = pre_flame_r*multi
            write_flames_g[num] = pre_flame_g*multi
            write_flames_b[num] = pre_flame_b*multi
        
        frame_r = np.uint8(np.max(write_flames_r,axis=0)[::-1,:]*255)
        frame_g = np.uint8(np.max(write_flames_g,axis=0)[::-1,:]*255)
        frame_b = np.uint8(np.max(write_flames_b,axis=0)[::-1,:]*255)
        
        frame = np.stack([frame_r,frame_g,frame_b],axis=2)
        img = Image.fromarray(frame)
        img.save(img_output_folder+str(index).zfill(file_num_zfill)+'.jpg')


def min_max_Normalization(array_1D):
    if np.min(array_1D) == np.max(array_1D):
        output_data = np.zeros_like(array_1D)
    else:
        output_data = (array_1D - np.min(array_1D))/(np.max(array_1D) - np.min(array_1D))
    return output_data    

def revers_point(point,width):
    if width/2 < point:
        output = -width/2 + point
    else:
        output = width/2 + point
    return output

def lowpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2   #ナイキスト周波数
    wp = fp / fn  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn  #ナイキスト周波数で阻止域端周波数を正規化

    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y  

def main():
    #wave_data_file_name = r"audio2image_test_music.wav"
    #wave_data_file_name = r"test_stereo_diff.wav"
    wave_data_file_name = r"cosmic_dark_ambient.wav"
    wave_data_folder = r""
    #img_output_folder = r""
    #img_output_folder = r""
    img_output_folder = r""
    fps = 30
    #au_display_sample_num = 1024*32
    au_display_sample_num = int(44100*0.8)
    delay_time = 0.5
    display_freq_max = 1500
    #output_width = 1280
    #output_height = 720
    #output_width = 640
    #output_height = 360
    output_width = 640
    output_height = 260
    #output_width = 160
    #output_height = 120
    file_num_zfill = 10

    make_frame_cir(wave_data_file_name,
                       wave_data_folder,
                       img_output_folder,
                       fps,
                       au_display_sample_num,
                       display_freq_max,
                       delay_time,
                       output_width,
                       output_height,
                       file_num_zfill)
    
if __name__ == "__main__":   
    main()

