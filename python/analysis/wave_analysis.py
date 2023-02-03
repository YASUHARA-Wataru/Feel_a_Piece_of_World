# -*- coding: utf-8 -*-
"""
@author: YASUHARA Wataru

MIT License

"""

import numpy as np
from scipy import signal
from scipy import stats
from analysis import LoadWav
from typing import List
from dataclasses import dataclass,field

@dataclass
class wave_params_list:
    power: List[float] = field(default_factory=list)
    Freq_peak: List[float] = field(default_factory=list)
    demod_phase: List[float] = field(default_factory=list)
    var: List[float] = field(default_factory=list)
    skew: List[float] = field(default_factory=list)
    kurt: List[float] = field(default_factory=list)

@dataclass
class wave_params_np:
    power: np.ndarray = np.array([])
    Freq_peak: np.ndarray = np.array([])
    demod_phase: np.ndarray = np.array([])
    var: np.ndarray = np.array([])
    skew: np.ndarray = np.array([])
    kurt: np.ndarray = np.array([])


def calc_wave_params(wav_data,
                     au_data_time,
                     a_sr,fps,
                     au_display_sample_num):
    
    img_frame_times = np.arange(0,au_data_time[-1]+1/a_sr,1/fps)
    wav = wav_data
    wav_params = wave_params_list()

    for frame_index,img_frame_time in enumerate(img_frame_times):
        au_img_fit_ind = int(img_frame_time* a_sr)
        print("\r"+str(frame_index)+":"+str(au_img_fit_ind),end="")
        cut_index = np.arange(int(au_img_fit_ind)-int(au_display_sample_num),int(au_img_fit_ind),1)
        cut_index[cut_index < 0] = 0
        cut_index[cut_index >= au_data_time.shape[0]] = au_data_time.shape[0] - 1
        cut_index = cut_index.astype('int')
        
        FFT_wav = wav[cut_index]

        m_data = np.abs(signal.hilbert(FFT_wav))
        
        wav_params.var.append(np.var(m_data))

        if np.sum(m_data-np.mean(m_data)) < pow(10,-8):
            wav_params.skew.append(np.nan)
            wav_params.kurt.append(np.nan)
        else:
            wav_params.skew.append(stats.skew(m_data))
            wav_params.kurt.append(stats.kurtosis(m_data))


        FFT_data = np.fft.fft(FFT_wav)[:int(FFT_wav.shape[0]/2)]
        FFT_data[-1] = 0+0j
        
        Max_freq_ind = np.argmax(np.abs(FFT_data))

        freq = a_sr/2/(FFT_wav.shape[0]/2)*(Max_freq_ind)

        wav_params.Freq_peak.append(freq)
        #wav_params.demod_phase.append(np.angle(FFT_data[Max_freq_ind]))
        
        #"""
        mod_t = np.linspace(0, au_display_sample_num*(1/a_sr),au_display_sample_num)
        frame_init_time = au_img_fit_ind/a_sr
        demod_sin_wav = np.sin(2*np.pi*freq*mod_t)
        demod_cos_wav = np.cos(2*np.pi*freq*mod_t)
        I = FFT_wav[:au_display_sample_num]*demod_sin_wav
        Q = FFT_wav[:au_display_sample_num]*demod_cos_wav
        
        fp = 15*np.power(10,3)
        fs = 21*np.power(10,3)
        gpass = 3       #通過域端最大損失[dB]
        gstop = 40      #阻止域端最小損失[dB]
        
        if freq == 0:
            wav_params.demod_phase.append(0)
            wav_params.power.append(0)
        else:
            I_filted = lowpass(I,a_sr,fp,fs, gpass, gstop)
            Q_filted = lowpass(Q,a_sr,fp,fs, gpass, gstop)
            demod_phase_temp = np.arctan2(I_filted,Q_filted)[np.argmax(np.power(I_filted,2)+np.power(Q_filted,2))]
            init_phase = (np.mod(frame_init_time,1/freq)*freq)*2*np.pi
            if init_phase > np.pi:
                init_phase = -2*np.pi + init_phase
            wav_params.demod_phase.append(demod_phase_temp +init_phase)
            wav_params.power.append(np.sqrt(np.max(np.power(I_filted,2)+np.power(Q_filted,2))))
        #"""
    wav_params_np = wave_params_np()
    wav_params_np.power = np.array(wav_params.power)
    wav_params_np.Freq_peak = np.array(wav_params.Freq_peak)
    wav_params_np.demod_phase = np.mod((np.array(wav_params.demod_phase) + np.pi),(2*np.pi))
    
    wav_params_np.var = np.array(wav_params.var)
    wav_params_np.skew = np.array(wav_params.skew)  
    wav_params_np.kurt = np.array(wav_params.kurt)  
    
    return wav_params_np
    
def lowpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2   #ナイキスト周波数
    wp = fp / fn  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn  #ナイキスト周波数で阻止域端周波数を正規化

    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y  


def main():
    wave_data_file_name = r"test_stereo_diff.wav"
    wave_data_folder = r"test_data\\"
    fps = 30
    au_display_sample_num = int(44100*0.8)

    au_data,au_data_time,a_sr = LoadWav.load_wav(wave_data_folder+wave_data_file_name)
    
    ch1_wave_params = calc_wave_params(au_data[0],
                                    au_data_time,
                                    a_sr,fps,
                                    au_display_sample_num)  
    ch2_wave_params = calc_wave_params(au_data[1],
                                    au_data_time,
                                    a_sr,fps,
                                    au_display_sample_num)  


    print(ch1_wave_params.power)
    print(ch1_wave_params.Freq_peak)
    print(ch1_wave_params.demod_phase)
    print(ch1_wave_params.var)
    print(ch1_wave_params.skew)
    print(ch1_wave_params.kurt)

    print(ch2_wave_params.power)
    print(ch2_wave_params.Freq_peak)
    print(ch2_wave_params.demod_phase)
    print(ch2_wave_params.var)
    print(ch2_wave_params.skew)
    print(ch2_wave_params.kurt)

    
if __name__ == "__main__":   
    main()

