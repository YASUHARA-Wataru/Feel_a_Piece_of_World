# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 23:03:12 2022

@author: NavigateSafetyField
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def lowpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2   #ナイキスト周波数
    wp = fp / fn  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y  

sr = 44.1*pow(10,3)
time = np.arange(0,1,1/sr)
f_mod = 2000
f_remod = 2000
f_sig = 200
init_phase = np.pi/3
#init_phase = 0
display_ind = np.arange(0,2/f_sig*sr,1).astype("int")

sig_wave = np.sin(2*np.pi*f_sig*time)
mod_wave = np.sin(2*np.pi*f_mod*time+init_phase)
N = 1024
window = np.concatenate([np.hamming(N),np.zeros(mod_wave.shape[0]-N)],axis=0)


send_wave = sig_wave+mod_wave

FFT_result = np.fft.fft(send_wave)
FFT_result = np.abs(FFT_result[:int(send_wave.shape[0]/2)])/send_wave.shape[0]
FFT_freq = np.arange(int(FFT_result.shape[0]))*(sr/2)/(FFT_result.shape[0])
display_freq = sr/2
print(FFT_freq[1]-FFT_freq[0])
display_freq_ind = int(display_freq/(sr/2)*FFT_result.shape[0])

hilb_sig = signal.hilbert(send_wave)

remod_sin_wave = np.sin(2*np.pi*f_remod*time)
remod_cos_wave = np.cos(2*np.pi*f_remod*time)

I = send_wave * remod_sin_wave
Q = send_wave * remod_cos_wave

fp = f_sig*20
fs = f_sig*30
gpass = 3       #通過域端最大損失[dB]
gstop = 40      #阻止域端最小損失[dB]
 
I_filted = lowpass(I,sr,fp,fs, gpass, gstop)
Q_filted = lowpass(Q,sr,fp,fs, gpass, gstop)

Amp = np.sqrt(np.power(I_filted*2,2) + np.power(Q_filted*2,2))
Phase = np.arctan2(Q_filted,I_filted)
#Phase = np.arctan(Q_filted/I_filted)

plt.figure()
#plt.plot(FFT_freq[:display_freq_ind],FFT_result[:display_freq_ind],'.')
plt.plot(send_wave[display_ind])
plt.plot(I_filted[display_ind])
#plt.plot(Amp[display_ind])
#plt.plot(np.abs(hilb_sig[display_ind]))
plt.figure()
plt.plot(np.angle(hilb_sig[display_ind]))
plt.plot(Phase[display_ind])
#plt.plot(init_phase,'.')
plt.show()
print(str(Phase[np.argmax(Amp)])+':'+str(init_phase))