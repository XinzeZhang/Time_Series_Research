import numpy as np


def window_max(series, index, k):
    presequent = series[index-k:index]
    subsequent = series[index+1:index+k+1]

    window= presequent + subsequent
    return(max(window))

def window_min(series, index, k):
    presequent = series[index-k:index]
    subsequent = series[index+1:index+k+1]

    window= presequent + subsequent
    return(min(window))

def pivot_k_window(ts_data, k):
    # type of ts_data should be list
    ts_list=ts_data
    # get head-k and tail-k rid of time series
    search_list=ts_list[k:-k]
    ts_idx=list(range(k,len(ts_list)-k))
    # print(search_list)
    # print(ts_idx)
    peak_dic={}
    trough_dic = {}

    for idx, ts in zip(ts_idx,search_list):
        if ts > window_max(ts_list, idx, k):
            peak_dic[idx]=ts
        if ts < window_min(ts_list,idx,k):
            trough_dic[idx]=ts
    
    return peak_dic, trough_dic

data = np.load("./Data/Crude_Oil_Price/WTI.npz")
data = data["arr_0"]
# data type of .npz is numpy.array and need be transformed to list
data = data.tolist()

pivot_k_window(data,10)