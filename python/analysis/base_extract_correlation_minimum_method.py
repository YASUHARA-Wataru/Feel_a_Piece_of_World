# -*- coding: utf-8 -*-
"""
@author: YASUHARA WATARU

The MIT License (MIT)

"""
import numpy as np

def freq_analysis_1D(data):
    """
    Parameters
    ----------
    data : 1D-array
        1D-array Data.

    Raises
    ------
    Exception
        If Data is not 1D-array, raises error
        If Data contains negative value, raises error

    Returns
    -------
    stan_freq_num : TYPE
        Freqs value.

    """
    data = np.array(data)

    if np.min(data) < 0:
        raise Exception('data contains negative.')

    if data.flatten().shape[0] != len(list(data)):
        raise Exception('data is not 1D array.')

    data_num = data.shape[0]
    max_freqs = int(data_num/2)+1
    freq_nums = np.zeros(max_freqs-1,dtype='f')
    
    for freq in range(1,max_freqs):
        for i in range(data_num-freq-1):
            freq_nums[freq-1] += np.min([data[i],data[i+freq+1]])
            
    # normalize
    bin_nums = np.arange(data_num-1,data_num-max_freqs,-1)
    stan_freq_num = freq_nums/bin_nums

    return stan_freq_num
    

def freq_analysis_2D(data):
    """
    Parameters
    ----------
    data : 2D-array
        2D-array Data.

    Raises
    ------
    Exception
        If Data is not 2D-array, raises error
        If Data contains negative value, raises error

    Returns
    -------
    stan_freq_num : TYPE
        Freqs value.

    """

    data = np.array(data)

    if np.min(data) < 0:
        raise Exception('data contains negative.')

    if np.array(data.shape).shape[0] != 2:
        raise Exception('data is not 2D array.')

    data_dim1 = data.shape[0]
    data_dim2 = data.shape[1]

    freqs_rank1 = int(data_dim1/2)+1
    freqs_rank2 = int(data_dim2/2)+1
    freq_nums = np.zeros((freqs_rank1,freqs_rank2),dtype='f')
    

    for freq1 in range(freqs_rank1):
        for freq2 in range(freqs_rank2):            
            for i in range(data_dim1-freq1-1):
                for j in range(data_dim2-freq2-1):
                    interest1_data = data[i,j]
                    interest2_data = data[i+freq1,j+freq2]
                    interest3_data = data[i,j+freq2]
                    interest4_data = data[i+freq1,j+freq2]
                    interest_data = [interest1_data,
                                        interest2_data,
                                        interest3_data,
                                        interest4_data]
                    freq_nums[freq1,freq2] += np.min(interest_data)

    # normalize
    rank1_nums = np.arange(data_dim1,data_dim1-freqs_rank1,-1).reshape((freqs_rank1,1))
    rank2_nums = np.arange(data_dim2,data_dim2-freqs_rank2,-1).reshape((1,freqs_rank2))
    bin_nums = np.dot(rank1_nums,rank2_nums)
        
    stan_freq_num = freq_nums/bin_nums

    return stan_freq_num

def continuous_analisys_1D(data):
    """
    Parameters
    ----------
    data : TYPE
        1D-array Data.

    Raises
    ------
    Exception
        If Data is not 1D-array, raises error
        If Data contains negative value, raises error

    Returns
    -------
    stan_continuous_num : TYPE
        continuous num.

    """
    
    if np.min(data) < 0:
        raise Exception('data contains negative.')

    data = np.array(data)
    if data.flatten().shape[0] != len(list(data)):
        raise Exception('data is not 1D array.')

    data_num = data.shape[0]
    continuous_num = np.zeros_like(data)
    
    for continuous in range(0,data_num):
        for i in range(data_num-continuous):
            continuous_num[continuous] += np.min(data[i:i+continuous+1])
            

    # normalize
    bin_nums = np.arange(data_num,0,-1)
    stan_continuous_num = continuous_num/bin_nums

    return stan_continuous_num

def continuous_analisys_2D(data):
    """
    Parameters
    ----------
    data : 2D-array
        2D-array Data.
        If Data contains negative value, raises error

    Raises
    ------
    Exception
        If Data is not 2D-array, raises error
        If Data contains negative value, raises error

    Returns
    -------
    stan_continuous_num : TYPE
        continuous num.

    """
    data = np.array(data)

    if np.min(data) < 0:
        raise Exception('data contains negative.')

    if np.array(data.shape).shape[0] != 2:
        raise Exception('data is not 2D array.')

    data_dim1 = data.shape[0]
    data_dim2 = data.shape[1]
    continuous_num = np.zeros_like(data)

    for continuous1 in range(0,data_dim1):
        for continuous2 in range(0,data_dim2):
            for i in range(data_dim1-continuous1):
                for j in range(data_dim2-continuous2):
                    continuous_num[continuous1,continuous2] += np.min(data[i:i+continuous1+1,j:j+continuous2+1])

    # normalize
    rank1_nums = np.arange(data_dim1,0,-1).reshape((data_dim1,1))
    rank2_nums = np.arange(data_dim2,0,-1).reshape((1,data_dim2))
    bin_nums = np.dot(rank1_nums,rank2_nums)
    stan_continuous_num = continuous_num/bin_nums

    return stan_continuous_num

def any_base_analysis_1D_only_positive(data,base):
    """
    Parameters
    ----------
    data : TYPE
        1D-array Data
    base : TYPE
        1D-array base

    Raises
    ------
    Exception
        If Data is not 1D-array, raises error
        If Data contains negative value, raises error
        If base contains negative value, raises error
        If base is lager than Data, raises error

    Returns
    -------
    cor_result : TYPE
        DESCRIPTION.

    """
    data = np.array(data)
    base = np.array(base)

    if np.min(data) < 0:
        raise Exception('data contains negative.')

    if np.min(base) < 0:
        raise Exception('base can\'t contain negative values.')
 
    if data.flatten().shape[0] != len(list(data)) :
        raise Exception('data is not 1D array.')

    data_num = data.shape[0]
    base_num = base.shape[0]
    
    if data_num < base_num:
        raise Exception('base is too large.')
    
    cor_result = np.zeros(data_num - base_num,dtype='f')
    for i in range(data_num-base_num):
        data_cut = data[i:i+base_num]
        calc_index = base > 0
        cor_result[i] = np.min(data_cut[calc_index]*base[calc_index])
        
    return cor_result
            
def any_base_analysis_2D_only_positive(data,base):
    """
    Parameters
    ----------
    data : TYPE
        2D-array Data
    base : TYPE
        2D-array base

    Raises
    ------
    Exception
        If Data is not 2D-array, raises error
        If Data contains negative value, raises error
        If base contains negative value, raises error
        If base is lager than Data, raises error

    Returns
    -------
    cor_result : TYPE
        cor_result

    """
    data = np.array(data)
    base = np.array(base)

    if np.min(data) < 0:
        raise Exception('data contains negative.')

    if np.min(base) < 0:
        raise Exception('base can\'t contain negative values.')
 
    if np.array(data.shape).shape[0] != 2:
        raise Exception('data is not 2D array.')

    data_num_dim1 = data.shape[0]
    data_num_dim2 = data.shape[1]
    base_num_dim1 = base.shape[0]
    base_num_dim2 = base.shape[1]
    
    if data_num_dim1 < base_num_dim1:
        raise Exception('base Dim1 is too large.')
    if data_num_dim2 < base_num_dim2:
        raise Exception('base Dim2 is too large.')

    
    cor_result = np.zeros((data_num_dim1 - base_num_dim1,data_num_dim2 - base_num_dim2),dtype='f')
    for i in range(data_num_dim1-base_num_dim1):
        for j in range(data_num_dim2-base_num_dim2):
            data_cut = data[i:i+base_num_dim1,j:j+base_num_dim2]
            calc_index = base > 0
            cor_result[i,j] = np.min(data_cut[calc_index]*base[calc_index])
        
    return cor_result
        
def any_base_analysis_1D(data,base):
    """
    Parameters
    ----------
    data : TYPE
        1D-array Data
    base : TYPE
        1D-array base

    Raises
    ------
    Exception
        If Data is not 1D-array, raises error
        If Data contains negative value, raises error
        If base contains negative value, raises error
        If base is lager than Data, raises error

    Returns
    -------
    cor_result : TYPE
        DESCRIPTION.

    """
    data = np.array(data)
    base = np.array(base)

 
    if data.flatten().shape[0] != len(list(data)) :
        raise Exception('data is not 1D array.')

    data_num = data.shape[0]
    base_num = base.shape[0]
    
    if data_num < base_num:
        raise Exception('base is too large.')
    
    cor_result = np.zeros(data_num - base_num,dtype='f')
    for i in range(data_num-base_num):
        data_cut = data[i:i+base_num]
        calc_index = base != 0 # maybe no problem using float
        cor_result[i] = np.min(data_cut[calc_index]*base[calc_index])
        
    return cor_result
            
def any_base_analysis_2D(data,base):
    """
    Parameters
    ----------
    data : TYPE
        2D-array Data
    base : TYPE
        2D-array base

    Raises
    ------
    Exception
        If Data is not 2D-array, raises error
        If Data contains negative value, raises error
        If base contains negative value, raises error
        If base is lager than Data, raises error

    Returns
    -------
    cor_result : TYPE
        cor_result

    """
    data = np.array(data)
    base = np.array(base)
 
    if np.array(data.shape).shape[0] != 2:
        raise Exception('data is not 2D array.')

    data_num_dim1 = data.shape[0]
    data_num_dim2 = data.shape[1]
    base_num_dim1 = base.shape[0]
    base_num_dim2 = base.shape[1]
    
    if data_num_dim1 < base_num_dim1:
        raise Exception('base Dim1 is too large.')
    if data_num_dim2 < base_num_dim2:
        raise Exception('base Dim2 is too large.')

    
    cor_result = np.zeros((data_num_dim1 - base_num_dim1,data_num_dim2 - base_num_dim2),dtype='f')
    for i in range(data_num_dim1-base_num_dim1):
        for j in range(data_num_dim2-base_num_dim2):
            data_cut = data[i:i+base_num_dim1,j:j+base_num_dim2]
            calc_index = base != 0 # maybe no problem using float
            cor_result[i,j] = np.min(data_cut[calc_index]*base[calc_index])
        
    return cor_result


def main():

    # test freq_analysis_1D
    value = np.zeros(100)
    value[::5] = 1
    freq_nums = freq_analysis_1D(value)
    print('freq_nums:' + str(freq_nums))
    
    # test_freq_analysis_2D
    value = np.zeros((40,30))
    value[::5,:] = 1
    value[:,::5] = 1
    freq_nums = freq_analysis_2D(value)
    print(freq_nums)

    # test_continuous_analysis_1D
    value = np.zeros(50)
    value[10:30] = 1
    continuous_num = continuous_analisys_1D(value)
    print(continuous_num)

    # test_continuous_analysis_2D
    value = np.zeros((50,50))
    value[0:40,0:20] = 1
    value[0:50,0:20] += 1
    continuous_num = continuous_analisys_2D(value)
    print(continuous_num)

    # test any_base_analysis_1D
    value = np.zeros(100)
    base = np.array([1,0,0,1,0,1])
    value[::5] = 1
    value[::3] += 1
    cor_result = any_base_analysis_1D_only_positive(value,base)
    print('cor_result:' + str(cor_result))

    # test any_base_analysis_2D
    value = np.zeros((100,100))
    base = np.array([[1,0,1,0],
                     [0,0,0,0],
                     [1,0,1,0],
                     [0,0,0,1]])
    value[:4,:4] = base
    cor_result = any_base_analysis_2D_only_positive(value,base)
    print('cor_result:' + str(cor_result))

    # test any_base_analysis_1D
    value = np.zeros(100)
    base = np.array([1,0,0,-1,0,1])
    value[:6] = base
    print(value)
    print(base)
    cor_result = any_base_analysis_1D(value,base)
    np.savetxt('text1.txt',cor_result)
    print('cor_result:' + str(cor_result))

    # test any_base_analysis_2D
    value = np.zeros((100,100))
    base = np.array([[1,0,1,0],
                     [0,0,0,0],
                     [1,0,-1,0],
                     [0,0,0,1]])
    value[:4,:4] = base
    cor_result = any_base_analysis_2D(value,base)
    print('cor_result:' + str(cor_result))


if __name__ == "__main__":
    main()
