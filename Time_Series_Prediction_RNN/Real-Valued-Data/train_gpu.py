from model_gpu import *

import torch.nn as nn

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from numpy import concatenate

import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import time

from _data_process import *

if __name__ == '__main__':
    #------------------------------------------------------------------------
    # load dataset
    series = read_csv('chinese_oil_production.csv', header=0,
                      parse_dates=[0], index_col=0, squeeze=True)

    # transfer the dataset to array
    raw_values = series.values
    ts_values_array = np.array(raw_values)
    set_length = len(ts_values_array)

    # transform data to be stationary
    dataset = difference(raw_values, 1)

    # creat dataset train, test
    ts_look_back = 12
    dataset = create_dataset(dataset, look_back=ts_look_back)

    # split into train and test sets
    train_size = int(dataset.shape[0] * 0.8)
    diff_length = dataset.shape[0]
    test_size = diff_length - train_size
    train, test = dataset[0:train_size], dataset[train_size:]

    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    # train_scaled :: shape:[train_num,seq_len] which meams [batch, input_size]
    # test_scaled :: shape:[train_num,seq_len] which meams [batch, input_size]

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # ---------------------------------------------------------------------------------------
    # load data and make training set
    train_input_scaled = train_scaled[:, :-1,np.newaxis]
    train_input = Variable(torch.from_numpy(
        train_input_scaled), requires_grad=False)
    

    train_target_scaled = train_scaled[:, 1:]
    train_target = Variable(torch.from_numpy(
        train_target_scaled), requires_grad=False)

    test_input_scaled = test_scaled[:, :-1,np.newaxis]
    test_input = Variable(torch.from_numpy(
        test_input_scaled), requires_grad=False)
    

    test_target_scaled = test_scaled[:, 1:]
    test_target = Variable(torch.from_numpy(
        test_target_scaled), requires_grad=False)
    
    # ========================================================================================
    # hyper parameters
    Num_layers=2
    Num_iters = 6
    hidden_size=500
    print_every = 50
    plot_every = 1
    learning_rate = 0.1
    Optim_method='_SGD'

    GRU=GRUModel(input_dim=1, hidden_size=1, output_dim=1, num_layers=1, cell="GRU", num_iters=2, learning_rate = 0.01, print_interval=50, plot_interval=1)
    Train_h_state=GRU.initHidden(train_input)
    Test_h_state=GRU.initHidden(test_input)
    # define the loss
    criterion = nn.MSELoss()
    # use LBFGS/SGD as optimizer since we can load the whole data to train
    # optimizer = optim.LBFGS(seq.parameters(), lr=learning_rate)
    if Optim_method=='_SGD':
        optimizer = optim.SGD(self.parameters(), lr=self.Learn_rate)
    if Optim_method=='_Adam':
        optimizer = optim.Adam(self.parameters(), lr=self.Learn_rate)

        # Initialize timer
    time_tr_start = time.time()

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # # compute the MSE and record the loss
    # def closure():
    #     optimizer.zero_grad()
    #     prediction, Train_h_state = GRU(train_input,Train_h_state)
    #     loss = criterion(prediction, train_target)
    #     global plot_loss_total
    #     global print_loss_total
    #     plot_loss_total += loss.data[0]
    #     print_loss_total += loss.data[0]
    #     loss.backward()
    #     return loss

    # begin to train
    for iter in range(1, Num_iters + 1):
        # optimizer.step(closure)
        prediction, Train_h_state = GRU(train_input,Train_h_state)
        Train_h_state=Variable(Train_h_state.data)

        loss = criterion(prediction, train_target)
        plot_loss_total += loss.data[0]
        print_loss_total += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.8f' % (timeSince(time_tr_start, iter / Num_iters),
                                         iter, iter / Num_iters * 100, print_loss_avg))
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    plot_loss(plot_losses, Fig_name='L'+str(Num_layers)+'_H'+str(hidden_size)+'_I'+str(Num_iters)+Optim_method+'_Loss')

    #---------------------------------------------------------------------------------------
    # begin to forcast
    print('Forecasting Testing Data')

    # make a one-step forecast
    def forecast(input,GRU_h_state):
        pre, h_state = GRU(input,GRU_h_state)
        # pre = pre.cpu()
        pre = pre.data.numpy()
        return pre

    # get train_result
    Y_train = forecast(train_input,Train_h_state)
    Y_train = Y_train[:, -1]
    # inverse the train pred
    Y_train = invert_scale(scaler, train_input_scaled, Y_train)
    Y_train = inverse_train_difference(raw_values, Y_train, ts_look_back)

    # get test_result
    Y_pred = forecast(test_input,Test_h_state)
    Y_pred = Y_pred[:, -1]
    Y_target = test_target_scaled[:, -1]

    # get prediction loss
    MSE_loss = nn.MSELoss()
    Y_pred_torch = Variable(torch.from_numpy(Y_pred), requires_grad=False)
    Y_target_torch = Variable(torch.from_numpy(Y_target), requires_grad=False)
    MSE_pred = MSE_loss(Y_pred_torch, Y_target_torch)
    MSE_pred = MSE_pred.data.numpy()
    # inverse the test pred
    Y_pred = invert_scale(scaler, test_input_scaled, Y_pred)
    Y_pred = inverse_test_difference(
        raw_values, Y_pred, train_size, ts_look_back)

    # # print forecast
    # for i in range(len(test)):
    #     print('Predicted=%f, Expected=%f' % ( y_pred[i], raw_values[-len(test)+i]))

    plot_result(TS_values=ts_values_array, Train_value=Y_train,
                Pred_value=Y_pred, Loss_pred=MSE_pred, Fig_name='L'+str(Num_layers)+'_H'+str(hidden_size)+'_I'+str(Num_iters)+'_Prediction')

