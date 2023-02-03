# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 02:03:49 2022

@author: NavigateSafetyFiled
"""

import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

aud_output_folder = r""
aud_name = r"test_stereo_diff.wav"

sr = 44.1*pow(10,3)
tone_length = 2
amp_div = 4

output_tones = [-9,-7,-5,-4,-2,0,2,3]

standard_freq = 440
"""
do = standard_freq*pow(2,-9/12)
re = standard_freq*pow(2,-7/12)
mi = standard_freq*pow(2,-5/12)
fa = standard_freq*pow(2,-4/12)
so = standard_freq*pow(2,-2/12)
ra = standard_freq
si = standard_freq*pow(2,2/12)
"""

amplitude = np.iinfo(np.int16).max/amp_div
t_material = np.arange(0., tone_length, 1/sr)
tone_materials = []

for degree in output_tones:
    on = amplitude * np.sin(2. * np.pi * standard_freq*pow(2,degree/12) * t_material)
    on = on*np.hanning(on.shape[0])
    off = np.zeros_like(on)
    tone_materials.append(np.concatenate([on,off]))

conect_materials = np.array(tone_materials).reshape(len(tone_materials)*t_material.shape[0]*2)
ch1_sig = np.concatenate([conect_materials,np.zeros(int(t_material.shape[0]/2))],axis=0)
ch2_sig = np.concatenate([np.zeros(int(t_material.shape[0]/2)),conect_materials],axis=0)
#ch1_sig = np.zeros_like(ch1_sig)

#ch1_sig = conect_materials
#ch2_sig = conect_materials

sig = np.array([ch1_sig,ch2_sig]).transpose()

#fig = plt.figure()
#plt.plot(sig[:,0])
#plt.show()

print(sig.shape)
write(aud_output_folder+aud_name,int(sr),sig.astype(np.int16))