from __future__ import print_function
import torch
import torch.nn as nn
# from torch.autograd import Variable
import torch.optim as optim

from models.model_gpu import RNNModel
# from pandas import DataFrame
# from pandas import Series
# from pandas import concat
# from pandas import read_csv
# from pandas import datetime

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import numpy as np
from numpy import concatenate, atleast_2d

import matplotlib.ticker as ticker
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_process._data_process import create_dataset,plot_forecasting_result
from _definition import pivot_k_window, MDPP
from typing import List

import argparse
# Training settings
# ==============================================================================
parser = argparse.ArgumentParser(description='PyTorch Time Series Forecasting after Landmarks')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='hidden size for training (default: 64)')
parser.add_argument('--num_layers', type=int, default=1, metavar='N',
                    help='layers for training (default: 1)')
parser.add_argument('--cell', type=str, default='RNN', metavar='S',
                    help='cell types for training (default: RNN)')
parser.add_argument('--num_iters', type=int, default=101, metavar='N',
                    help='layers for training (default: 100)')                    
parser.add_argument('--optim_method', type=str, default='Adam', metavar='S',
                    help='optim_method  for training (default: Adam)')
parser.add_argument('--learning_rate', type=int, default=0.001, metavar='N',
                    help='learning_rate for training (default: 0.001)')
parser.add_argument('--print_interval', type=int, default=50, metavar='N',
                    help='print_interval for training (default: 50)')
parser.add_argument('--plot_interval', type=int, default=1, metavar='N',
                    help='plot_interval for training (default: 1)')

if __name__ == '__main__':
    args = parser.parse_args()

    # load dataset
    dirs = "./Data/Crude_Oil_Price/WTI.npz"
    # dirs = "./Data/Residential_Load/Residential_Load_hour.npz"
    temp = np.load(dirs)
    load_data = temp["arr_0"].tolist()
    # np.savez(dirs,load_data)
    ts_values_array = np.array(load_data)
    set_length = len(ts_values_array)

    k_windows = 5
    # peak_dic, trough_dic=pivot_k_window(load_data, k_windows)
    distance = 5
    percentage = 0.15
    marks_dic = MDPP(load_data, distance, percentage, k_windows)

    marks_range: List[int] = []
    marks_value: List[float] = []

    for idx in marks_dic:
        marks_range.append(idx)
        marks_value.append(marks_dic[idx])

    # -------------------------------------------------------------------
    input_dim = 7
    dataset = create_dataset(ts_values_array, look_back=input_dim)
    dataset_idx = create_dataset(np.arange(set_length), look_back=input_dim)
    # split into train and test sets
    train_size = int(dataset.shape[0] * 0.8) + input_dim
    train_scope = np.arange(train_size)
    # test_size = dataset.shape[0] - train_size
    test_scope = np.arange(train_size, set_length)
    # -------------------------------------------------------------------
    # divide the ts_values to train section and test section
    ts_train = ts_values_array[:train_size].copy()
    ts_test = ts_values_array[train_size:].copy()
    ts_train_idx = np.arange(set_length)[:train_size].copy()
    ts_test_idx = np.arange(set_length)[train_size:].copy()
    # -------------------------------------------------------------------
    # divide the ts_values to train sets and test sets
    train, test = dataset[0:train_size], dataset[train_size:]
    train_idx, test_idx = dataset_idx[0:train_size], dataset_idx[train_size:]
    # sample landmarks from test sets
    look_ahead = 4  # attention! look_ahead should be smaller than or equal to k_windows
    test_samples_idx = np.arange(test.shape[0])
    ts_test_marks_idx = list()
    # get mark index in test section
    for mark_idx in marks_range:
        if mark_idx in ts_test_idx:
            ts_test_marks_idx.append(mark_idx)

    test_marks_set = list()
    test_marks_set_idx = list()
    for mark in ts_test_marks_idx:
        for idx in test_samples_idx:
            sample_idx = test_idx[idx, :]
            if sample_idx[input_dim-1] == mark:
                for step in np.arange(look_ahead):
                    test_marks_set.append(test[idx+step, :])
                    test_marks_set_idx.append(test_idx[idx+step, :])

    test_marks_set_idx=np.array(test_marks_set_idx)
    test_marks_set_idx = np.unique(test_marks_set_idx,axis=0)
    test_marks_set_idx=np.array(test_marks_set_idx)

    test_marks_set=np.array(test_marks_set)
    test_marks_set = np.unique(test_marks_set,axis=0)
    test_marks_set=np.array(test_marks_set)
    train=np.array(train)

    # data shape should be (batch,len-ts,input-dim)
    train_input = atleast_2d(train[:,:-1])[:, :, np.newaxis]
    train_target = train[:,-1].reshape(train.shape[0],1)[:, :, np.newaxis]
    # --
    train_target_plot = train[:,-1].flatten().tolist()
    train_section = train_idx[:,-1].flatten().tolist()
    # # --
    # test_input = atleast_2d(test_marks_set[:,:-1])[:, :, np.newaxis]
    # test_target = test_marks_set[:,-1].reshape(test_marks_set.shape[0],1)[:, :, np.newaxis]
    # # --
    # test_target_plot = test_marks_set[:,-1].flatten().tolist()
    # test_section = test_marks_set_idx[:,-1].flatten().tolist()
    # --
    test_input = atleast_2d(test[:,:-1])[:, :, np.newaxis]
    test_target = test[:,-1].reshape(test.shape[0],1)[:, :, np.newaxis]
    # --
    test_target_plot = test[:,-1].flatten().tolist()
    test_section = test_idx[:,-1].flatten().tolist()

    # data shape should be (batch,len-ts,input-dim)
    train_input = torch.from_numpy(
        train_input).float()
    train_target = torch.from_numpy(
        train_target).float()
    # --
    test_input = torch.from_numpy(
        test_input).float()
    test_target = torch.from_numpy(
        test_target).float()

# ========================================================================================
    # hyper parameters
    # RNN_Cell:
    # 'GRU' or 'RNN'
    # # Optim_method:
    # 'SGD' or 'Adam' or 'RMSprop' or 'Adadelta' or 'Adagrad' or 'SparseAdam' or 'Adamax' or 'ASGD'

    # benchmark 1.1 rnn h100 i1120
    
    RNN_Demo = RNNModel(input_dim=1,
                        hidden_size=args.hidden_size,
                        output_dim=1,
                        num_layers=args.num_layers,
                        cell=args.cell,
                        num_iters=args.num_iters,
                        optim_method=args.optim_method,
                        learning_rate=args.learning_rate,
                        print_interval=args.print_interval,
                        plot_interval=args.plot_interval).cuda()
    # ========================================================================================
    Save_Road = './Results/COP'
    RNN_Demo.fit(train_input, train_target, save_road=Save_Road)

    # RNN_Demo.fit(train_input, train_target,View_interval)
    # save the model
    # model_save_road='./Model/Model' + '_L' + str(Num_layers) + '_H' + str(Hidden_size) + '_I' + str(Num_iters)+Optim_method+'.pkl'
    # torch.save(RNN_Demo,model_save_road)

    # RNN_Demo=Model_ViewList[0]
    # Train_ViewList = Model_ViewList[1]
    # ---------------------------------------------------------------------------------------
    # begin to forcast
    print('\n------------------------------------------------')
    print('Forecasting Testing Data')
    print('------------------------------------------------')

    Y_train = RNN_Demo.predict(train_input)
    train_pred = Y_train[:, -1, :]

    # get test_result
    Y_pred = RNN_Demo.predict(test_input)
    test_pred = Y_pred[:, -1, :]
    # Y_target = test_target

    # get prediction loss
    MSE_loss = nn.MSELoss()
    test_pred_torch = torch.from_numpy(
        test_pred).float()
    Y_target_torch = test_target[:,0,:]
    MSE_pred = MSE_loss(test_pred_torch, Y_target_torch)
    MSE_pred = MSE_pred.data.numpy()
    RMSE_pred = np.sqrt(MSE_pred)

    # # print forecast
    # for i in range(len(test)):
    #     print('Predicted=%f, Expected=%f' % ( y_pred[i], raw_values[-len(test)+i]))

    plot_fig_name = Save_Road+'_Pred_'+args.cell + '_L' + \
        str(args.num_layers) + '_H' + str(args.hidden_size) + \
        '_E' + str(args.num_iters)+'_'+args.optim_method

    train_target_plot = train_target.data.numpy()[:, -1, :].flatten()
    train_pred_plot=train_pred.flatten()
    test_target_plot = test_target.data.numpy()[:, 0, :].flatten()
    test_pred_plot=test_pred_torch.data.numpy().flatten()

    # get section of training and test sequence   
    test_scope = test_section
    train_scope = train_section


    plt.figure(figsize=(20, 5))
    plt.title(
        'Forecasting Future Values for Time Series', fontsize=12)
    plt.title('RMSE of Prediction: %(rmse).3e' %
              {'rmse': RMSE_pred}, loc='right', fontsize=10)
    plt.xlabel('Input Sequence', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.plot(train_scope, train_target_plot, 'c-', label='Training Target', linewidth=1)
    plt.plot(train_scope, train_pred_plot, 'm-', label='Training Result', linewidth=1)
    plt.plot(test_idx[:,-1], test[:,-1], 'b-.', label='Test Target', linewidth=1)
    plt.plot(test_scope, test_pred_plot, 'r-.', label='Test Result', linewidth=1)
    plt.plot(marks_range, marks_value, 'yo', label='ts_marks')
    plt.legend(loc='upper right')
    plt.savefig(plot_fig_name + '.png')
    '''                           
    # transform data to be stationary
    diff = difference(load_data, 1)

    # create differece_dataset x,y
    dataset = diff.values
    dataset = create_dataset(dataset, look_back=1)

    # split into train and test sets
    train_size = int(dataset.shape[0] * 0.8)
    train_scope=np.arange(train_size)
    # test_size = dataset.shape[0] - train_size
    test_scope= np.arange(train_size,set_length)
    #-------------------------------------------------------------------
    # divide the ts_values to train set and test set
    ts_train=ts_values_array[:train_size].copy()
    ts_test=ts_values_array[train_size:].copy()
    #--------------------------------------------------------------------
    #divide the ts_values_diff to train set and test set
    train, test = dataset[0:train_size], dataset[train_size:]
    ts_train_diff=train[:,:1]
    ts_test_diff=test[:,:1]
    ts_test_diff=np.append(ts_test_diff,dataset[-1,-1])
    #--------------------------------------------------------------------
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    # divided the train_set and test_set
    ts_trian_scaled=train_scaled[:,:1]
    ts_test_scaled=test_scaled[:,:1]
    ts_test_scaled=np.append(ts_test_scaled,test_scaled[-1,-1])
    
    # =====================================================================
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # plt.figure(figsize=(10,15))
    plt.figure(figsize=(15, 5))
    # locate the sublabel
    # draw the train set and test set
    # plt=plt.subplot(111)
    # plt.set_xticks(np.arange(0,set_length,10))
    plt.plot(train_scope, ts_train, 'k', label='ts_train', linewidth=1.5)
    plt.plot(test_scope, ts_test, 'r', label='ts_test', linewidth=1.5)
    # plt.plot(peak_range,peak_value,'c^',label='ts_peak')
    # plt.plot(trough_range,trough_value,'mv',label='ts_trough')
    plt.plot(marks_range, marks_value, 'yo', label='ts_marks')
    for key in marks_dic:
        # show_mark='('+str(key)+',%s)' %(marks_dic[key])
        show_mark = str(key)
        plt.annotate(show_mark, xy=(
            key, marks_dic[key]), fontsize=9, color='y')
    # plt.minorticks_on()
    # plt.grid(which='both')
    plt.legend(loc='upper right')
    plt.title('Values for Time Sequences')
    plt.xlabel('Time Sequence')
    plt.ylabel('Value')


    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ax2=plt.subplot(312,sharex=plt)
    ax2.plot(train_scope, ts_train_diff,'g',label='ts_train_diff',linewidth = 1.5)
    ax2.plot(test_scope, ts_test_diff,'r:',label='ts_test_diff',linewidth = 1.5)
    # ax2.minorticks_on()
    ax2.grid(which='both')
    ax2.legend(loc='upper right')
    ax2.set_title('Values_difference for Time Sequences')
    plt.xlabel('Time Sequence')
    plt.ylabel('Difference')
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ax3=plt.subplot(313,sharex=plt)
    ax3.plot(train_scope, ts_trian_scaled,'b',label='ts_train_diff_scaled',linewidth = 1.5)
    ax3.plot(test_scope, ts_test_scaled,'r:',label='ts_test_diff_scaled',linewidth = 1.5)

    # ax3.minorticks_on()
    ax3.grid(which='both')
    ax3.legend(loc='upper right')
    ax3.set_title('Values_difference_scaled for Time Sequences')
    plt.xlabel('Time Sequence')
    plt.ylabel('Scaled Difference')
 
    # plt.subplots_adjust(hspace=0.75)
    # plt.savefig('WTI_visualization.png')
    plt.show()
    '''