# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:11:55 2022

@author: YASUHARA_WORK
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot

def freq_cnt(data):
    data_num = data.shape[0]
    freqs = np.arange(2,int(data_num/2))
    freq_nums = np.zeros_like(freqs,dtype='f')
    freq1 = np.min(data)
    data = data - freq1
    
    for index,freq in enumerate(freqs):
        filt = np.zeros(freq*2)
        freq_index = np.arange(0,freq*2,freq)
        filt[freq_index] = 1
        for i in range(data_num-freq):
            end_ind = i + freq*2
            filt_end_ind = freq*2
            if end_ind > data_num:
                filt_end_ind = freq*2 - (end_ind - data_num)
                end_ind = data_num
            cor_data = data[i:end_ind]*filt[:filt_end_ind]
            freq_nums[index] += np.min(cor_data[freq_index])
    # """
    # 倍数補正

    # 素因数分解したときの1を除いた要素の数
    complted_multi_dict = {2:1,  # 2: 1*2 (mutliに1はつかわない)
                           3:1,  # 3: 1*3
                           4:2,  # 4: 1*4 2*2
                           5:1,  # 5: 1*5
                           6:3,  # 6: 1*6 2*3 3*2
                           7:1,  # 7: 1*7
                           8:2,  # 8: 1*8 2*4 4*2  8:3だが修正?
                           9:2,  # 9: 1*9 3*3
                           10:3, # 10: 1*10 2*5 5*2
                           11:1, # 11: 1*11
                           12:2, # 12: 1*12 2*6 3*4 4*3 6*2 12:5だが修正
                           13:1, # 13: 1*13
                           14:3, # 14: 1*14 2*7 7*2
                           15:3, # 15: 1*15 3*5 5*3
                           16:2, # 16: 1*16 2*8 4*4 8*2 16:4だが修正
                           17:1,
                           18:2, # 18:5だが修正
                           19:1,
                           20:2, # 20:5だが修正
                           21:3,
                           22:3,
                           23:1,
                           24:2, # 24:7だが修正
                           }
    com_m_value = np.zeros_like(freq_nums)
    com_multi_cnt = freq_nums.copy()

    for index,freq in enumerate(freqs):
        multi_freqs = np.arange(freq,int(data_num/2),freq)[1:]
        multis = (multi_freqs/freq).astype('i')
        com_m_value_each = np.zeros_like(freq_nums,dtype='f')
        com_m_temp = np.zeros_like(freq_nums,dtype='f')
        for multi,multi_freq in zip(multis,multi_freqs):
            com_filt = np.zeros(freq*2*multi)
            #freq_index_com = np.arange(0,freq*2*multi,freq)
            freq_index_com = np.arange(0,freq*multi+1,freq)
            com_filt[freq_index_com] = 1
            cnt = 0
            com_index = np.where(freqs==multi_freq)[0][0]
            for i in range(data_num-(multi_freq)):
                end_com_ind = i + freq*2*multi
                filt_end_com_ind = freq*2*multi
                if end_com_ind > data_num:
                    filt_end_com_ind = freq*2*multi - (end_com_ind - data_num)
                    freq_index_com[freq_index_com > filt_end_com_ind-1] = 0
                    end_com_ind = data_num

                cor_com_data = data[i:end_com_ind]*com_filt[:filt_end_com_ind]

                if np.min(cor_com_data[freq_index_com]) > 0:
                    cnt += 1
                  
                com_m_temp[com_index] += np.min(cor_com_data[freq_index_com])
                
            com_m_value_each[com_index] = com_m_temp[com_index]
            com_m_value[com_index] += com_m_temp[com_index]-com_m_temp[com_index]*(-1+complted_multi_dict[multi])
            com_multi_cnt[com_index] -= com_m_temp[com_index]-com_m_temp[com_index]*(-1+complted_multi_dict[multi])


    # 正規化
    bin_nums = np.zeros_like(freqs)
    for index,freq in enumerate(freqs):
        bin_nums[index] = data.shape[0]-freq+1
        
    stan_freq_num = freq_nums/bin_nums
    stan_com_multi_cnt = com_multi_cnt/bin_nums
    
    return freqs,freq_nums,stan_freq_num,com_multi_cnt,stan_com_multi_cnt


#num_of_time = 94
num_of_time = 90
#num_of_time = 100
time_sieries_data = np.zeros(num_of_time)
time_index = np.arange(num_of_time)
freq_come_person = np.array([3])
freq_num_come_persons = np.array([1])
freq4plot = 1/freq_come_person

freq_indexs=[]
for freq in freq_come_person:
    freq_indexs.append(np.arange(0,num_of_time,freq))

for freq_index,freq_num_come_person in zip(freq_indexs,freq_num_come_persons):
    time_sieries_data[freq_index] = time_sieries_data[freq_index] + freq_num_come_person


test_csv_name = r"test_data\\daily_views.csv"
df = pd.read_csv(test_csv_name)
test_data = np.array(df['views'])

plt.figure()
#plt.plot(time_sieries_data)
plt.plot(test_data)


#freqs,freq_nums,stan_freq_num,com_value,stan_com_value = freq_cnt(time_sieries_data)
freqs,freq_nums,stan_freq_num,com_value,stan_com_value = freq_cnt(test_data)

plt.figure()
#autocorrelation_plot(pd.Series(time_sieries_data))
autocorrelation_plot(pd.Series(test_data))

print(np.sum(test_data))
plt.figure()
plt.plot(freqs,freq_nums,'.')
#plt.ylim(0,20)
plt.figure()
plt.plot(freqs,stan_freq_num)
#plt.ylim(0,7)
plt.figure()
plt.plot(freqs,com_value)
plt.figure()
plt.plot(freqs,stan_com_value)
plt.show()


