# -*- coding: utf-8 -*-
"""
@author: YASUHARA Wataru

MIT License

"""

import numpy as np
from PIL import Image
from analysis import wave_analysis

def make_frame_cir_stereo_smooth(ch1_params:wave_analysis.wave_params_np,
                                 ch2_params:wave_analysis.wave_params_np,
                                 img_output_folder,
                                 fps,
                                 au_display_sample_num,
                                 display_freq_max,
                                 delay_time,
                                 output_width,
                                 output_height,
                                 file_num_zfill):
    
    ch1_data = ch1_params.power
    ch2_data = ch2_params.power
    ch1_power = np.ones_like(ch1_params.power)*0.1
    ch2_power = np.ones_like(ch2_params.power)*0.1
    ch1_Freq_peak = ch1_params.Freq_peak
    ch2_Freq_peak = ch2_params.Freq_peak
    ch1_demod_phase = ch1_params.demod_phase
    ch2_demod_phase = ch2_params.demod_phase
    ch1_var = ch1_params.var
    ch2_var = ch2_params.var
    ch1_skew = ch1_params.skew
    ch2_skew = ch2_params.skew
    ch1_kurt = ch1_params.kurt
    ch2_kurt = ch2_params.kurt

    ch1_skew = np.nan_to_num(ch1_skew,nan=np.nanmean(ch1_skew))  
    ch2_skew = np.nan_to_num(ch2_skew,nan=np.nanmean(ch2_skew))  
    ch1_kurt = np.nan_to_num(ch1_kurt,nan=np.nanmean(ch1_kurt))  
    ch2_kurt = np.nan_to_num(ch2_kurt,nan=np.nanmean(ch2_kurt))  

    ch1_Freq_peak = (ch1_Freq_peak)/display_freq_max
    ch2_Freq_peak = (ch2_Freq_peak)/display_freq_max
    ch1_vars = min_max_Normalization(ch1_var)
    ch2_vars = min_max_Normalization(ch2_var)
    
    ch1_skews = min_max_Normalization(ch1_skew)
    ch2_skews = min_max_Normalization(ch2_skew)
    ch1_kurts = min_max_Normalization(ch1_kurt)
    ch2_kurts = min_max_Normalization(ch2_kurt)  
    
    max_size_per_width = 8
    min_size_per_width = 100
    ch1_size_params = ch1_power*output_width/max_size_per_width+output_width/min_size_per_width
    ch2_size_params = ch2_power*output_width/max_size_per_width+output_width/min_size_per_width
    ch1_r_posi_params = np.log10(min_max_Normalization(ch1_data)+1)/np.log10(2)*output_height/2
    ch2_r_posi_params = np.log10(min_max_Normalization(ch2_data)+1)/np.log10(2)*output_height/2
    ch1_the_posi_params = np.array(ch1_demod_phase)
    ch2_the_posi_params = np.array(ch2_demod_phase)

    pre_flames_r = np.zeros((int(fps*delay_time),output_height,output_width))
    pre_flames_g = np.zeros((int(fps*delay_time),output_height,output_width))
    pre_flames_b = np.zeros((int(fps*delay_time),output_height,output_width))
    
    for index in range(ch1_size_params.shape[0]):
        if index ==0:
            pre_index = 0
        else:
            pre_index = index - 1
        
        intgral_num = 100
        ch1_diff_r = np.linspace(ch1_r_posi_params[pre_index],ch1_r_posi_params[index],intgral_num)
        ch1_diff_the = ch1_the_posi_params[pre_index] - ch1_the_posi_params[index] 
        ch1_diff_the2 = revers_angle(ch1_the_posi_params[pre_index]) - revers_angle(ch1_the_posi_params[index])
        ch1_diff_the_comp = np.min([np.abs(ch1_diff_the),np.abs(ch1_diff_the2)])
        ch1_diff_the_calc = ch1_diff_the_comp / intgral_num 
        
        ch2_diff_r = np.linspace(ch2_r_posi_params[pre_index],ch2_r_posi_params[index],intgral_num)
        ch2_diff_the = ch2_the_posi_params[pre_index] - ch2_the_posi_params[index] 
        ch2_diff_the2 = revers_angle(ch2_the_posi_params[pre_index]) - revers_angle(ch2_the_posi_params[index])
        ch2_diff_the_comp = np.min([np.abs(ch2_diff_the),np.abs(ch2_diff_the2)])
        ch2_diff_the_calc = ch2_diff_the_comp / intgral_num

        smooth_deg = 5
        if ch1_diff_the_comp < smooth_deg/180*np.pi:
            ch1_move_dis = np.abs(ch1_r_posi_params[pre_index] - ch1_r_posi_params[index])
        else:
            ch1_move_dis = np.sum(ch1_diff_r*ch1_diff_the_calc)

        if ch2_diff_the_comp < smooth_deg/180*np.pi:
            ch2_move_dis = np.abs(ch2_r_posi_params[pre_index] - ch2_r_posi_params[index])
        else:
            ch2_move_dis = np.sum(ch2_diff_r*ch2_diff_the_calc)
        
        interpre_multi = 1/2
        ch1_interpre_num = int(ch1_move_dis*interpre_multi)
        ch2_interpre_num = int(ch2_move_dis*interpre_multi)
        print("\r" + str(index)+":"+str(int(ch1_interpre_num+ch2_interpre_num)),end="")


        if ch1_interpre_num == 0:
            ch1_interpre_num = 1
        if ch2_interpre_num == 0:
            ch2_interpre_num = 1

        ch1_x_index = np.tile(np.tile(np.arange(0,output_width),(output_height,1)),(ch1_interpre_num,1,1)).transpose(1,2,0)
        ch1_y_index = np.tile(np.tile(np.arange(0,output_height),(output_width,1)).T,(ch1_interpre_num,1,1)).transpose(1,2,0)
        ch2_x_index = np.tile(np.tile(np.arange(0,output_width),(output_height,1)),(ch2_interpre_num,1,1)).transpose(1,2,0)
        ch2_y_index = np.tile(np.tile(np.arange(0,output_height),(output_width,1)).T,(ch2_interpre_num,1,1)).transpose(1,2,0)

        ch1_size_param = np.linspace(ch1_size_params[pre_index],ch1_size_params[index],ch1_interpre_num)
        ch2_size_param = np.linspace(ch2_size_params[pre_index],ch2_size_params[index],ch2_interpre_num)

        ch1_r_posi_param = np.linspace(ch1_r_posi_params[pre_index],ch1_r_posi_params[index],ch1_interpre_num)
        ch2_r_posi_param = np.linspace(ch2_r_posi_params[pre_index],ch2_r_posi_params[index],ch2_interpre_num)
                
        ch1_diff_the = ch1_the_posi_params[pre_index] - ch1_the_posi_params[index] 
        ch1_diff_the2 = revers_angle(ch1_the_posi_params[pre_index]) - revers_angle(ch1_the_posi_params[index])
        ch2_diff_the = ch2_the_posi_params[pre_index] - ch2_the_posi_params[index] 
        ch2_diff_the2 = revers_angle(ch2_the_posi_params[pre_index]) - revers_angle(ch2_the_posi_params[index])
        
        if np.abs(ch1_diff_the) > np.abs(ch1_diff_the2):
            ch1_the_posi_param = np.linspace(revers_angle(ch1_the_posi_params[pre_index]),revers_angle(ch1_the_posi_params[index]),ch1_interpre_num)
            ch1_the_posi_param = revers_angle1D(ch1_the_posi_param)
        else:
            ch1_the_posi_param = np.linspace(ch1_the_posi_params[pre_index],ch1_the_posi_params[index],ch1_interpre_num)


        if np.abs(ch2_diff_the) > np.abs(ch2_diff_the2):
            ch2_the_posi_param = np.linspace(revers_angle(ch2_the_posi_params[pre_index]),revers_angle(ch2_the_posi_params[index]),ch2_interpre_num)
            ch2_the_posi_param = revers_angle1D(ch2_the_posi_param)
        else:
            ch2_the_posi_param = np.linspace(ch2_the_posi_params[pre_index],ch2_the_posi_params[index],ch2_interpre_num)

        ch1_y_posi_param = output_height/2 + ch1_r_posi_param*np.cos(ch1_the_posi_param)
        ch2_y_posi_param = output_height/2 + ch2_r_posi_param*np.cos(ch2_the_posi_param)
        ch1_x_posi_param = output_width/2 + ch1_r_posi_param*np.sin(ch1_the_posi_param)
        ch2_x_posi_param = output_width/2 + ch2_r_posi_param*np.sin(ch2_the_posi_param)
        
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
            
        frame_r = np.uint8(np.max(write_flames_r,axis=0)*255)
        frame_g = np.uint8(np.max(write_flames_g,axis=0)*255)
        frame_b = np.uint8(np.max(write_flames_b,axis=0)*255)
        
        frame = np.stack([frame_r,frame_g,frame_b],axis=2)
        img = Image.fromarray(frame)
        img.save(img_output_folder+str(index).zfill(file_num_zfill)+'.jpg')


def min_max_Normalization(array_1D):
    if np.min(array_1D) == np.max(array_1D):
        output_data = np.zeros_like(array_1D)
    else:
        output_data = (array_1D - np.min(array_1D))/(np.max(array_1D) - np.min(array_1D))
    return output_data    

def revers_angle(angle):    
    if angle < 0:
        output_angle = np.pi + angle
    else:
        output_angle = -np.pi + angle
    return output_angle

def revers_angle1D(angle1D):
    output_angle = np.zeros_like(angle1D)
    output_angle[angle1D < 0] = np.pi + angle1D[angle1D < 0]
    output_angle[angle1D >= 0] = -np.pi + angle1D[angle1D >= 0]
    return output_angle


def revers_point(point,width):
    if width/2 < point:
        output = -width/2 + point
    else:
        output = width/2 + point
    return output

def main():
    pass

if __name__ == "__main__":   
    main()

