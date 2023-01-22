# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 11:15:29 2023

@author: YASUHARA Wataru

The MIT License (MIT)
"""
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import base_extract_correlation_minimum_method as becmm


# 1D data test
test_csv_name = r"analysis\\test_data\\daily_views.csv"
#test_csv_name = r"test_data\\daily_views_alias.csv"

df = pd.read_csv(test_csv_name)
value = df['views']
data_date = pd.to_datetime(df['date'])

date_diff=[]
for date in data_date:
    date_diff.append((date - data_date.min()).days)

# functionn of test
freq_nums = becmm.freq_analysis_1D(value)
continous = becmm.continuous_analisys_1D(value)
base = [1,0,0]
base_ana = becmm.any_base_analysis_1D(value, base)

# embed for plot
freq_nums_emb4plot = np.zeros_like(value,dtype='f')
freq_nums_emb4plot[:freq_nums.shape[0]] = freq_nums
base_ana_emb4plot = np.zeros_like(value,dtype='f')
base_ana_emb4plot[:base_ana.shape[0]] = base_ana

# make figure
fig, ax = plt.subplots(4, 1, sharex="all", figsize=(6, 5),tight_layout=True)
fig.suptitle("base_extract_correlation_minimum_method 1D analysis")
ax[0].set_title('Data')
ax[0].plot(date_diff,value)
ax[1].set_title('freq analysis')
ax[1].bar(date_diff,freq_nums_emb4plot)
ax[2].set_title('contious analysis')
ax[2].bar(date_diff,continous)
# x is phase
ax[3].set_title('any base analysis')
ax[3].bar(date_diff,base_ana_emb4plot)
plt.xticks(date_diff[::5])
fig.autofmt_xdate()


# 2D data test
#img_name = r"analysis\\test_data\\lena.jpg"
img_name = r"analysis\\test_data\\test_span_3.png"
#img_name = r"analysis\\test_data\\test_span_32.png"
test_image = np.array(Image.open(img_name),dtype='f')

# functionn of test
freq_nums = becmm.freq_analysis_2D(test_image)
continous = becmm.continuous_analisys_2D(test_image)
base = [[1,0,0,1],
        [0,0,0,0],
        [0,0,0,0],
        [1,0,0,1]]
base_ana = becmm.any_base_analysis_2D(test_image, base)

# embed for plot
freq_nums_emb4plot = np.zeros_like(test_image,dtype='f')
freq_nums_emb4plot[:freq_nums.shape[0],:freq_nums.shape[1]] = freq_nums
base_ana_emb4plot = np.zeros_like(test_image,dtype='f')
base_ana_emb4plot[:base_ana.shape[0],:base_ana.shape[1]] = base_ana

# make figure
fig, ax = plt.subplots(2, 2, figsize=(5, 5),tight_layout=True)
fig.suptitle("base_extract_correlation_minimum_method 2D analysis")
ax[0][0].set_title('Data')
ax[0][0].pcolor(test_image,cmap='gray')
ax[1][0].set_title('freq analysis')
ax[1][0].pcolor(freq_nums,cmap='gray')
ax[0][1].set_title('contious analysis')
ax[0][1].pcolor(continous,cmap='gray')
# x is phase
ax[1][1].set_title('any base analysis')
ax[1][1].pcolor(base_ana_emb4plot,cmap='gray')

plt.show()
