# -*- coding: utf-8 -*-
"""
@author: YASUHARA Wataru
"""

import wave
import numpy as np

def load_wav(wavfilepath):
    # load wav file
    wf   = wave.open(wavfilepath, "r")
    # get wav parameters
    sr        = wf.getframerate() # sampling freq
    n_ch      = wf.getnchannels() # ch number
    n_frames  = wf.getnframes()   # num of audio frame
    n_bytes   = wf.getsampwidth() # byte per bit
    
    audiobuffer = wf.readframes(n_frames)
    wf.close()

    if   n_bytes == 1: # 8-bit uint
        norm = 2**7
        data = np.frombuffer(audiobuffer, dtype=np.uint8).astype(np.float32)
        data = data.copy() - norm
    elif n_bytes == 2: # 16-bit int
        data = np.frombuffer(audiobuffer, dtype=np.int16).astype(np.float32)
        norm = 2**15

    n_samples = data.shape[0] // n_ch
    x = np.zeros((n_ch, n_samples), dtype=np.float32)
    for k in range(n_ch):
        x[k] = data[k::n_ch]

    x /= norm
    print(wavfilepath)
    print(np.min(x), np.max(x))
    print('SR:' + str(sr))
    print('n_ch:' + str(n_ch))
    print('n_frames:' + str(n_frames))
    print('n_bytes:' + str(n_bytes))
    data_time = np.arange(0,1/sr*n_frames,1/sr)
    return x, data_time,sr


def main():
    wave_data_file_name = r"test_stereo_diff.wav"
    wave_data_folder = r"test_data\\"
        
    au_data,au_data_time,a_sr = load_wav(wave_data_folder+wave_data_file_name)
    print(au_data_time)
    print(au_data)
    print(a_sr)

            
if __name__ == "__main__":   
    main()

