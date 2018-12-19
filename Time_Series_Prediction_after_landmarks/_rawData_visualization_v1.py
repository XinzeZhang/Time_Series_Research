from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from numpy import concatenate

import matplotlib.ticker as ticker
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _definition import pivot_k_window, MDPP
from typing import List

import os
# load data from .npz
# the data in ./Data/Residential_Load/ has 3 columns, meters_date, meters_time and load_data.


def load_data(filename):
    temp = np.load(filename)
    return temp["arr_0"], temp["arr_1"], temp["arr_2"]

# convert an array of values into a dataset matrix


def create_dataset(dataset, look_back=1):
    # dataset = np.insert(dataset, [0] * look_back, 0)
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    dataY = np.array(dataY)
    dataY = np.reshape(dataY, (dataY.shape[0], 1))
    dataset = np.concatenate((dataX, dataY), axis=1)
    return dataset

# create a differenced series


def difference(dataset, interval=1):
    diff = list()
    # diff.append(0)
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value


def inverse_difference(history, yhat, interval=1):
    ori = list()
    for i in range(len(yhat)):
        value = yhat[i]+history[-interval+i]
        ori.append(value)
    return Series(ori).values

# scale train and test data to [-1, 1]


def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value


def invert_scale(scaler, ori_array, pred_array):
    # reshape the array to 2D
    pred_array = pred_array.reshape(pred_array.shape[0], 1)
    ori_array = ori_array.reshape(ori_array.shape[0], 1)
    # maintain the broadcast shape with scaler
    pre_inverted = concatenate((ori_array, pred_array), axis=1)
    inverted = scaler.inverse_transform(pre_inverted)
    # extraction the pred_array_inverted
    pred_array_inverted = inverted[:, -1]
    return pred_array_inverted

# remove the former n samples from the list


def remove_former(input_list, n):
    data_list = input_list[n:]
    return data_list


if __name__ == '__main__':
    # load dataset
    dirs = "./Data/Crude_Oil_Price/WTI.npz"
    # dirs = "./Data/Residential_Load/Residential_Load_hour.npz"
    temp = np.load(dirs)
    load_data = temp["arr_0"].tolist()
    # np.savez(dirs,load_data)
    ts_array = np.array(load_data)
    set_length = len(ts_array)
    ts_idx = list(range(set_length))

    k_windows = 4
    # peak_dic, trough_dic=pivot_k_window(load_data, k_windows)
    marks_dic = MDPP(load_data, 12, 0.20)

    # peak_range: List[int] = []
    # peak_value: List[float] = []
    # trough_range: List[int] = []
    # trough_value: List[float] = []
    marks_range: List[int] = []
    marks_value: List[float] = []

    # for idx in peak_dic:
    #     peak_range.append(idx)
    #     peak_value.append(peak_dic[idx])
    # for idx in trough_dic:
    #     trough_range.append(idx)
    #     trough_value.append(trough_dic[idx])
    for idx in marks_dic:
        marks_range.append(idx)
        marks_value.append(marks_dic[idx])

    #remove the former 4 turning points of the series
    marks_range = remove_former(marks_range, 4)
    marks_value = remove_former(marks_value, 4)

    for point_idx, i in zip(marks_range, range(len(marks_range))):
        print(point_idx, i+1)
        dirs = "./Data/Crude_Oil_Price/WTI_"+str(i+1)+"_"+str(point_idx)
        if not os.path.exists(dirs):
            os.mkdir(dirs)
        training_samples = ts_array[:point_idx+1]
        # shape:(N,T) s.t. N+T-1= length
        training_set = create_dataset(training_samples, look_back=12)
        np.savez(dirs+"/trainSet.npz",training_set)
        # training_idx = ts_idx[:point_idx+1]
        test_samples=ts_array[point_idx+1-12:point_idx+1+12]
        test_set=create_dataset(test_samples, look_back=12)
        np.savez(dirs+"/testSet.npz",test_set)
        # test_idx=ts_idx[point_idx+1-12:point_idx+1+12]
        print()
    exit()
    #=====================================================================
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # plt.figure(figsize=(10,15))
    plt.figure(figsize=(15, 5))
    # locate the sublabel
    # draw the train set and test set
    # plt=plt.subplot(111)
    # plt.set_xticks(np.arange(0,set_length,10))
    plt.plot(np.arange(set_length), ts_array,
             'k', label='raw series', linewidth=1)

    plt.plot(marks_range, marks_value, 'r.', label='turning points')
    # for key in marks_dic:
    #     # show_mark='('+str(key)+',%s)' %(marks_dic[key])
    #     show_mark = str(key)
    #     plt.annotate(show_mark, xy=(
    #         key, marks_dic[key]), fontsize=9, color='y')
    # plt.minorticks_on()
    # plt.grid(which='both')
    plt.legend(loc='upper right')
    # plt.title('West Texas Intermediate Crude Oil Price')
    plt.xlabel('Time Sequence')
    plt.ylabel('Value')
    plt.savefig('WTI_visualization.png')
    plt.show()

    '''
    #=====================================================================
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    plt.figure(figsize=(180,60))
    # plt.figure()
    # locate the sublabel
    # draw the train set and test set
    ax1=plt.subplot(311)
    # ax1.set_xticks(np.arange(0,set_length,10))
    ax1.plot(train_scope, ts_train,'k',label='ts_train',linewidth = 1.0)
    ax1.plot(test_scope, ts_target,'r',label='ts_target',linewidth = 1.0)
    # ax1.minorticks_on()
    # ax1.grid(which='both')
    ax1.legend(loc='upper right')
    ax1.set_title('Values for Time Sequences')
    plt.xlabel('Time Sequence' )
    plt.ylabel('Value')
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ax2=plt.subplot(312,sharex=ax1)
    ax2.plot(train_scope, ts_train_diff,'g',label='ts_train_diff',linewidth = 1.0)
    ax2.plot(test_scope, ts_test_diff,'r:',label='ts_target_diff',linewidth = 1.0)
    # ax2.minorticks_on()
    ax2.grid(which='both')
    ax2.legend(loc='upper right')
    ax2.set_title('Values_difference for Time Sequences')
    plt.xlabel('Time Sequence')
    plt.ylabel('Difference')
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ax3=plt.subplot(313,sharex=ax1)
    ax3.plot(train_scope, ts_trian_scaled,'b',label='ts_train_diff_scaled',linewidth = 1.0)
    ax3.plot(test_scope, ts_test_scaled,'r:',label='ts_target_diff_scaled',linewidth = 1.0)
    # ax3.minorticks_on()
    ax3.grid(which='both')
    ax3.legend(loc='upper right')
    ax3.set_title('Values_difference_scaled for Time Sequences')
    plt.xlabel('Time Sequence')
    plt.ylabel('Scaled Difference')

    plt.subplots_adjust(hspace=0.75)
    plt.show()
    # plt.savefig(str(meters_id)+'_hour_load_visualization.png')
    '''
