import numpy as np


def window_max(series, index, k):
    presequent = series[index-k:index]
    subsequent = series[index+1:index+k+1]

    window = presequent + subsequent
    return(max(window))


def window_min(series, index, k):
    presequent = series[index-k:index]
    subsequent = series[index+1:index+k+1]

    window = presequent + subsequent
    return(min(window))


def pivot_k_window(ts_data, k):
    # type of ts_data should be list
    ts_list = ts_data
    # get head-k and tail-k rid of time series
    search_list = ts_list[k:-k]
    ts_idx = list(range(k, len(ts_list)-k))
    # print(search_list)
    # print(ts_idx)
    peak_dic = {}
    trough_dic = {}

    for idx, ts in zip(ts_idx, search_list):
        if ts > window_max(ts_list, idx, k):
            peak_dic[idx] = ts
        if ts < window_min(ts_list, idx, k):
            trough_dic[idx] = ts

    return peak_dic, trough_dic


def MDPP(ts_data, distance, percentage):
    # https://ieeexplore.ieee.org/document/839385/ "Landmarks: a new model for similarity-based pattern querying in time series databases"
    # type of ts_data should be list
    ts_list = ts_data
    ts_idx = list(range(len(ts_list)))
    # pruning the begin and end
    search_list = ts_list[1:-1]
    ts_idx = ts_idx[1:-1]
    marks_dic = {}
    # first step: get 1-order landmarks
    for idx, ts in zip(ts_idx, search_list):
        if ts > window_max(ts_list, idx, 1):
            marks_dic[idx] = ts
        if ts < window_min(ts_list, idx, 1):
            marks_dic[idx] = ts
    # second step: remove landmarks under MDPP
    print(marks_dic)
    list_idxs=list(marks_dic.keys())
    
    i=0
    while i < list(range(len(list_idxs)))[-1]:
        _idx=list_idxs[i]
        _idx_R1= list_idxs[i+1]
        _value = marks_dic[_idx]
        _value_R1= marks_dic[_idx_R1]
        d= _idx_R1-_idx
        p=2.0*abs(_value_R1-_value)/(abs(_value_R1)+abs(_value))
        # print(d,p)
        if (d < distance and p < percentage)==True:
            print(_idx,_idx_R1,marks_dic[_idx],marks_dic[_idx_R1])
            del marks_dic[_idx]
            del marks_dic[_idx_R1]
            i+=1
        i+=1            
    print(marks_dic)
    return marks_dic


if __name__ == '__main__':
    data = np.load("./Data/Crude_Oil_Price/WTI.npz")
    data = data["arr_0"]
    # data type of .npz is numpy.array and need be transformed to list
    data = data.tolist()

    MDPP(data,4,0.05)