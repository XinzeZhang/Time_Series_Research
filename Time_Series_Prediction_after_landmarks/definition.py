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

def DP(ts_dic,ts_idx,idx,distance,percentage):
    marks_dic=ts_dic
    list_idxs=ts_idx
    
    i=idx
    _idx=list_idxs[i]
    _idx_R1= list_idxs[i+1]
    _idx_R2= list_idxs[i+2]
    _value = marks_dic[_idx]
    _value_R1= marks_dic[_idx_R1]
    _value_R2= marks_dic[_idx_R2]
    d_i1= _idx_R1-_idx
    p_i1=2.0*abs(_value_R1-_value)/(abs(_value_R1)+abs(_value))
    d_i2=_idx_R2-_idx_R1
    p_i2=2.0*abs(_value_R2-_value_R1)/(abs(_value_R2)+abs(_value_R1))

    return d_i1,p_i1,d_i2,p_i2


def MDPP(ts_data, distance, percentage,k=2):
    # https://ieeexplore.ieee.org/document/839385/ "Landmarks: a new model for similarity-based pattern querying in time series databases"
    # type of ts_data should be list
    ts_list = ts_data
    ts_idx = list(range(len(ts_list)))
    # pruning the begin and end
    search_list = ts_list[k:-k]
    ts_idx = ts_idx[k:-k]
    marks_dic = {}
    # first step: get 1-order landmarks
    for idx, ts in zip(ts_idx, search_list):
        if ts > window_max(ts_list, idx, k):
            marks_dic[idx] = ts
        if ts < window_min(ts_list, idx, k):
            marks_dic[idx] = ts
    # second step: remove landmarks under MDPP
    # print(marks_dic)
    list_idxs=list(marks_dic.keys())
    
    i=0
    while i < list(range(len(list_idxs)))[-k]:
        # _idx=list_idxs[i]
        # _idx_R1= list_idxs[i+1]
        # _value = marks_dic[_idx]
        # _value_R1= marks_dic[_idx_R1]
        # d= _idx_R1-_idx
        # p=2.0*abs(_value_R1-_value)/(abs(_value_R1)+abs(_value))

        d_i1,p_i1,d_i2,p_i2 = DP(marks_dic,list_idxs,i,distance,percentage)
        # print(d,p)
        if (d_i1 < distance and p_i1 < percentage)==True:
            # print(_idx,_idx_R1,marks_dic[_idx],marks_dic[_idx_R1])
            del marks_dic[list_idxs[i]]
            # if ( p_i2 < percentage)==True:
            del marks_dic[list_idxs[i+1]]
            i+=1
        i+=1            
    # print(marks_dic)
    peak_dic = {}
    trough_dic = {}
    
    for key in marks_dic:
        if ts_list[key-1] < ts_list[key]:
            peak_dic[key]= ts_list[key]
        if ts_list[key-1] > ts_list[key]:
            trough_dic[key] = ts_list[key]
    return marks_dic


if __name__ == '__main__':
    data = np.load("./Data/Crude_Oil_Price/WTI.npz")
    data = data["arr_0"]
    # data type of .npz is numpy.array and need be transformed to list
    data = data.tolist()

    MDPP(data,4,0.05)