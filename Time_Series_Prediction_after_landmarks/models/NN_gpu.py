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
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import time

from data_process._data_process import timeSince,plot_loss,plot_train

import matplotlib
matplotlib.use('agg') # avoiding Invalid DISPLAY variable
import matplotlib.pyplot as plt
from matplotlib import animation


def weights_init(m):
    if isinstance(m,nn.RNN):
        # randome initialize the nn weight with STD
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param,std=0.015)
            elif 'weight' in name:
                nn.init.normal_(param,std=0.015)

    if isinstance(m,nn.GRU):
        # randome initialize the nn weight with STD
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param,std=0.015)
            elif 'weight' in name:
                nn.init.normal_(param,std=0.015)

    if isinstance(m,nn.LSTM):
        # randome initialize the nn weight with STD
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param,std=0.015)
            elif 'weight' in name:
                nn.init.normal_(param,std=0.015)

    if isinstance(m,nn.Linear):
        # alternative random initialize way
        m.weight.data.normal_(0, 0.015)
        m.bias.data.normal_(0, 0.015)


# ---------------
# Base model
# ---------------
class BaseModel(nn.Module):
    def __init__(self, input_dim=1, hidden_size=150, output_dim=1, num_layers=1, cell="GRU"):
        super(BaseModel, self).__init__()
        self.Input_dim = input_dim
        self.Output_dim = output_dim
        self.Hidden_Size = hidden_size
        self.Num_layers = num_layers
        if cell == "RNN":
            self.Cell = nn.RNN(input_size=self.Input_dim, hidden_size=self.Hidden_Size,
                               num_layers=self.Num_layers, dropout=0.0,
                               nonlinearity="relu", batch_first=True,)
        if cell == "LSTM":
            self.Cell = nn.LSTM(input_size=self.Input_dim, hidden_size=self.Hidden_Size,
                                num_layers=self.Num_layers, dropout=0.0,
                                batch_first=True, )
        if cell == "GRU":
            self.Cell = nn.GRU(input_size=self.Input_dim, hidden_size=self.Hidden_Size,
                               num_layers=self.Num_layers, dropout=0.0,
                               batch_first=True,)
        if cell == "Linear":
            self.Cell = nn.Linear(in_features=self.Input_dim, out_features=self.Hidden_Size
                               )                        
        self.fc = nn.Linear(self.Hidden_Size, self.Output_dim)


# ---------------
# RNN or GRU inherit base model
# ---------------
class rnnModel(BaseModel):
    '''
    RNN/GRU inherit base model
    '''
    def __init__(self, input_dim, hidden_size, output_dim, num_layers, cell, num_iters, optim_method='SGD', learning_rate=0.001, print_interval=50, plot_interval=1, view_interval=1):
        super(rnnModel, self).__init__(
            input_dim, hidden_size, output_dim, num_layers, cell)
        self.cell_name = cell
        self.Print_interval = print_interval
        self.Plot_interval = plot_interval
        self.Num_iters = num_iters
        self.Optim_method = optim_method
        self.Learn_rate = learning_rate
        print('================================================')
        # print(self.Cell)
        print(self.cell_name+'_L'+str(self.Num_layers) + '_H' +
              str(self.Hidden_Size)+'_I'+str(self.Num_iters)+'_'+self.Optim_method)
        print('================================================\n')

    def forward(self, input, h_state):
        # input: shape[batch,time_step,input_dim]
        # h_state: shape[layer_num*direction,batch,hidden_size]
        # rnn_output: shape[batch,time_sequence_length,hidden_size]
        RNN_Output, h_state_n = self.Cell(input, h_state)
        FC_Outputs = []  # save all predictions

        for time_step in range(RNN_Output.size(1)):
            RNN_Output_time_step = RNN_Output[:, time_step, :]
            FC_Output_time_step = self.fc(RNN_Output_time_step)
            FC_Outputs.append(FC_Output_time_step)

        Outputs = torch.stack(FC_Outputs, dim=1)
        # Outputs = Outputs_T.squeeze(2).cuda()

        return Outputs, h_state_n

    def initHidden(self, input):
        batchSize = input.size(0)
        result = torch.empty(self.Num_layers * 1,
                             batchSize, self.Hidden_Size).float().cuda()
        result = nn.init.normal_(result,std=0.015)
        return result

    def fit(self, input, target, save_road='./Results/'):
        input = input.cuda()
        target = target.cuda()
        RNN_h_state = self.initHidden(input)
        criterion = nn.MSELoss()
        if self.Optim_method == 'ASGD':
            optimizer = optim.ASGD(self.parameters(), lr=self.Learn_rate)
        elif self.Optim_method == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.Learn_rate)
        elif self.Optim_method == 'RMSprop':
            optimizer = optim.RMSprop(self.parameters(), lr=self.Learn_rate)
        elif self.Optim_method == 'Adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=self.Learn_rate)
        elif self.Optim_method == 'Adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=self.Learn_rate)
        elif self.Optim_method == 'SparseAdam':
            optimizer = optim.Adagrad(self.parameters(), lr=self.Learn_rate)
        elif self.Optim_method == 'Adamax':
            optimizer = optim.Adamax(self.parameters(), lr=self.Learn_rate)
        elif self.Optim_method == 'SGD':
            optimizer = optim.SGD(
                self.parameters(), lr=self.Learn_rate, momentum=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

        # Initialize timer
        time_tr_start = time.time()

        plot_losses = []
        train_print_loss_total = 0  # Reset every print_every
        train_plot_loss_total = 0  # Reset every plot_every

        # plt.figure(1,figsize=(30,5))# continuously plot
        # plt.ion() # continuously plot

        # input_size=input.size(0)# continuously plot
        # time_period=np.arange(input_size)# continuously plot

        # begin to train
        for epoch in range(1, self.Num_iters + 1):
            scheduler.step()
            # input: shape[batch,time_step,input_dim]
            # h_state: shape[layer_num*direction,batch,hidden_size]
            # rnn_output: shape[batch,time_sequence_length,hidden_size]
            prediction, RNN_h_state = self.forward(input, RNN_h_state)
            RNN_h_state = RNN_h_state.data.cuda()
            prediction_2d = prediction[:, -1, :]
            # prediction=torch.reshape(prediction,(prediction.shape[0],1,1))
            target_2d = target[:, 0, :]
            loss = criterion(prediction_2d, target_2d)
            training_rmse = np.sqrt(loss.item())
            train_plot_loss_total += training_rmse
            train_print_loss_total += training_rmse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            # target_view=input[:,-1,:].data.numpy()# continuously plot
            # prediction_view=prediction[:,-1,:].data.numpy()
            # plt.plot(time_period,target_view.flatten(),'r-')
            # plt.plot(time_period,prediction_view.flatten(),'b-')
            # plt.draw();plt.pause(0.05)

            if epoch % self.Print_interval == 0:
                print_loss_avg = train_print_loss_total / self.Print_interval
                train_print_loss_total = 0
                print('%s (%d %d%%) ' % (timeSince(time_tr_start, epoch / self.Num_iters),
                                             epoch, epoch / self.Num_iters * 100))
                print('Training RMSE:  \t %.3e' % (print_loss_avg))
            if epoch % self.Plot_interval == 0:
                plot_loss_avg = train_plot_loss_total / self.Plot_interval
                plot_losses.append(plot_loss_avg)
                train_plot_loss_total = 0

        # Plot loss figure
        plot_loss(plot_losses, Fig_name=save_road+'Loss_'+self.cell_name+'_L'+str(self.Num_layers) +
                  '_H'+str(self.Hidden_Size)+'_I'+str(self.Num_iters)+'_'+self.Optim_method)
        print('\n------------------------------------------------')
        print('RNN Model finished fitting')
        print('------------------------------------------------')

        return self

    def predict(self, input):
        y_pred = self._predict(input)
        return y_pred

    def _predict(self,input):
        input = input.cuda()
        predict_h_state = self.initHidden(input)
        y_pred, predict_h_state = self.forward(input, predict_h_state)
        y_pred = y_pred.cpu().data.numpy()
        return y_pred

    def fit_validate(self, train_input, train_target, validate_input, validate_target, save_road='./Results/'):
        train_input = train_input.cuda()
        train_target = train_target.cuda()
        validate_input = validate_input.cuda()
        validate_target = validate_target.cuda()

        RNN_h_state = self.initHidden(train_input)
        validate_RNN_h_state = self.initHidden(validate_input)

        criterion = nn.MSELoss()
        if self.Optim_method == 'ASGD':
            optimizer = optim.ASGD(self.parameters(), lr=self.Learn_rate)
        elif self.Optim_method == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.Learn_rate)
        elif self.Optim_method == 'RMSprop':
            optimizer = optim.RMSprop(self.parameters(), lr=self.Learn_rate)
        elif self.Optim_method == 'Adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=self.Learn_rate)
        elif self.Optim_method == 'Adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=self.Learn_rate)
        elif self.Optim_method == 'SparseAdam':
            optimizer = optim.Adagrad(self.parameters(), lr=self.Learn_rate)
        elif self.Optim_method == 'Adamax':
            optimizer = optim.Adamax(self.parameters(), lr=self.Learn_rate)
        elif self.Optim_method == 'SGD':
            optimizer = optim.SGD(
                self.parameters(), lr=self.Learn_rate, momentum=0)

        # Initialize timer
        time_tr_start = time.time()

        training_losses = []
        validate_losses = []

        train_print_loss_total = 0  # Reset every print_every
        train_plot_loss_total = 0  # Reset every plot_every

        validate_print_loss_total = 0  # Reset every print_every
        validate_plot_loss_total = 0  # Reset every plot_every

        # plt.figure(1,figsize=(30,5))# continuously plot
        # plt.ion() # continuously plot

        # input_size=input.size(0)# continuously plot
        # time_period=np.arange(input_size)# continuously plot

        # begin to train
        for iter in range(1, self.Num_iters + 1):
            # input: shape[batch,time_step,input_dim]
            # h_state: shape[layer_num*direction,batch,hidden_size]
            # rnn_output: shape[batch,time_sequence_length,hidden_size]
            prediction, RNN_h_state = self.forward(train_input, RNN_h_state)
            RNN_h_state = RNN_h_state.data.cuda()
            loss = criterion(prediction, train_target)
            training_rmse = np.sqrt(loss.item())
            train_plot_loss_total += training_rmse
            train_print_loss_total += training_rmse
            optimizer.zero_grad()
            loss.backward()

            validate_prediction, validate_RNN_h_state_pred = self.forward(
                validate_input, validate_RNN_h_state)
            validate_loss = criterion(validate_prediction, validate_target)
            validate_rmse = np.sqrt(validate_loss.item())
            validate_print_loss_total += validate_rmse
            validate_plot_loss_total += validate_rmse

            optimizer.step()

            # target_view=input[:,-1,:].data.numpy()# continuously plot
            # prediction_view=prediction[:,-1,:].data.numpy()
            # plt.plot(time_period,target_view.flatten(),'r-')
            # plt.plot(time_period,prediction_view.flatten(),'b-')
            # plt.draw();plt.pause(0.05)

            if iter % self.Print_interval == 0:
                print_loss_avg = train_print_loss_total / self.Print_interval
                train_print_loss_total = 0

                validate_print_loss_avg = validate_print_loss_total / self.Print_interval
                validate_print_loss_total = 0

                print('%s (%d %d%%) ' % (timeSince(time_tr_start, iter / self.Num_iters),
                                         iter, iter / self.Num_iters * 100))
                print('Training RMSE:  \t %.3e\nValidating RMSE:\t %.3e' %
                      (print_loss_avg, validate_print_loss_avg))

            if iter % self.Plot_interval == 0:
                plot_loss_avg = train_plot_loss_total / self.Plot_interval
                training_losses.append(plot_loss_avg)
                train_plot_loss_total = 0

                validate_plot_loss_avg = validate_plot_loss_total / self.Plot_interval
                validate_losses.append(validate_plot_loss_avg)
                validate_plot_loss_total = 0

        # Plot loss figure
        # Plot RMSE Loss Figure
        training_losses = np.sqrt(training_losses)
        validate_losses = np.sqrt(validate_losses)
        plot_train(training_losses, validate_losses, Fig_title=self.cell_name+'_L'+str(self.Num_layers)+'_H'+str(self.Hidden_Size)+'_E'+str(self.Num_iters)+'_' +
                   self.Optim_method, Fig_name=save_road+'_Loss_'+self.cell_name+'_L'+str(self.Num_layers) + '_H'+str(self.Hidden_Size)+'_E'+str(self.Num_iters)+'_'+self.Optim_method)
        print('\n------------------------------------------------')
        print('RNN Model finished fitting')
        print('------------------------------------------------')

        return self

    def fit_view(self, input, target, view_interval, save_road='./Results/'):

        self.View_interval = view_interval
        input = input.cuda()
        target = target.cuda()
        RNN_h_state = self.initHidden(input)
        criterion = nn.MSELoss()
        if self.Optim_method == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.Learn_rate)
        if self.Optim_method == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.Learn_rate)

        # Initialize timer
        time_tr_start = time.time()

        plot_losses = []
        train_print_loss_total = 0  # Reset every print_every
        train_plot_loss_total = 0  # Reset every plot_every

        Predict_ViewList = []
        # begin to train
        for iter in range(1, self.Num_iters + 1):
            # input: shape[batch,time_step,input_dim]
            # h_state: shape[layer_num*direction,batch,hidden_size]
            # rnn_output: shape[batch,time_sequence_length,hidden_size]
            prediction, RNN_h_state = self.forward(input, RNN_h_state)
            RNN_h_state = RNN_h_state.data.cuda()
            loss = criterion(prediction, target)
            train_plot_loss_total += loss.item()
            train_print_loss_total += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % self.View_interval == 0:
                Predict_ViewList.append(prediction[:, -1, :].cpu().data)
            if iter % self.Print_interval == 0:
                print_loss_avg = train_print_loss_total / self.Print_interval
                train_print_loss_total = 0
                print('%s (%d %d%%) %.8f' % (timeSince(time_tr_start, iter / self.Num_iters),
                                             iter, iter / self.Num_iters * 100, print_loss_avg))
            if iter % self.Plot_interval == 0:
                plot_loss_avg = train_plot_loss_total / self.Plot_interval
                plot_losses.append(plot_loss_avg)
                train_plot_loss_total = 0

        # Plot loss figure
        plot_loss(plot_losses, Fig_name=save_road+'Loss_'+self.cell_name+'_L'+str(self.Num_layers) +
                  '_H'+str(self.Hidden_Size)+'_I'+str(self.Num_iters)+'_'+self.Optim_method)
        print('\n------------------------------------------------')
        print('RNN Model finished fitting')
        print('------------------------------------------------')

        return self, Predict_ViewList

# ---------------
# LSTM inherit base model
# ---------------
class lstmModel(BaseModel):
    '''
    LSTM inherit base model
    '''
    def __init__(self, input_dim, hidden_size, output_dim, num_layers, cell, num_iters, optim_method='SGD', learning_rate=0.1, print_interval=50, plot_interval=1, view_interval=1):
        super(lstmModel, self).__init__(
            input_dim, hidden_size, output_dim, num_layers, cell)
        self.cell_name = cell
        self.Print_interval = print_interval
        self.Plot_interval = plot_interval
        self.Num_iters = num_iters
        self.Optim_method = optim_method
        self.Learn_rate = learning_rate

    def forward(self, input, h_state, c_state):
        # input: shape[batch,time_step,input_dim]
        # h_state: shape[layer_num*direction,batch,hidden_size]
        # rnn_output: shape[batch,time_sequence_length,hidden_size]
        lstm_Output, (h_state_n, c_state_n) = self.Cell(input, (h_state, c_state))
        FC_Outputs = []  # save all predictions

        for time_step in range(lstm_Output.size(1)):
            lstm_Output_time_step = lstm_Output[:, time_step, :]
            FC_Output_time_step = self.fc(lstm_Output_time_step)
            FC_Outputs.append(FC_Output_time_step)

        Outputs = torch.stack(FC_Outputs, dim=1)
        # Outputs = Outputs_T.squeeze(2).cuda()

        return Outputs, h_state_n, c_state_n

    def initHidden(self, input):
        batchSize = input.size(0)
        result = torch.empty(self.Num_layers * 1,
                             batchSize, self.Hidden_Size).float().cuda()
        result = nn.init.normal_(result,std=0.015)
        return result

    def fit(self, input, target, save_road='./Results/'):
        print('================================================')
        # print(self.Cell)
        print(self.cell_name+'_L'+str(self.Num_layers) + '_H' +
              str(self.Hidden_Size)+'_I'+str(self.Num_iters)+'_'+self.Optim_method)
        print('================================================\n')
        input = input.cuda()
        target = target.cuda()
        lstm_h_state = self.initHidden(input)
        lstm_c_state = self.initHidden(input)
        criterion = nn.MSELoss()
        if self.Optim_method == 'SGD':
            optimizer = optim.SGD(
                self.parameters(), lr=self.Learn_rate, momentum=0)
        if self.Optim_method == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.Learn_rate)

        # Initialize timer
        time_tr_start = time.time()

        plot_losses = []
        train_print_loss_total = 0  # Reset every print_every
        train_plot_loss_total = 0  # Reset every plot_every

        # plt.figure(1,figsize=(30,5))# continuously plot
        # plt.ion() # continuously plot

        # input_size=input.size(0)# continuously plot
        # time_period=np.arange(input_size)# continuously plot

        # begin to train
        for iter in range(1, self.Num_iters + 1):
            # input: shape[batch,time_step,input_dim]
            # h_state: shape[layer_num*direction,batch,hidden_size]
            # rnn_output: shape[batch,time_sequence_length,hidden_size]
            prediction, lstm_h_state, lstm_c_state = self.forward(input, lstm_h_state,lstm_c_state)
            lstm_h_state = lstm_h_state.data.cuda()
            lstm_c_state = lstm_c_state.data.cuda()
            prediction_2d = prediction[:, -1, :]
            # prediction=torch.reshape(prediction,(prediction.shape[0],1,1))
            target_2d = target[:, 0, :]
            loss = criterion(prediction_2d, target_2d)
            training_rmse = np.sqrt(loss.item())
            train_plot_loss_total += training_rmse
            train_print_loss_total += training_rmse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # target_view=input[:,-1,:].data.numpy()# continuously plot
            # prediction_view=prediction[:,-1,:].data.numpy()
            # plt.plot(time_period,target_view.flatten(),'r-')
            # plt.plot(time_period,prediction_view.flatten(),'b-')
            # plt.draw();plt.pause(0.05)

            if iter % self.Print_interval == 0:
                print_loss_avg = train_print_loss_total / self.Print_interval
                train_print_loss_total = 0
                print('%s (%d %d%%) ' % (timeSince(time_tr_start, iter / self.Num_iters),
                                             iter, iter / self.Num_iters * 100))
                print('Training RMSE:  \t %.3e' % (print_loss_avg))
            if iter % self.Plot_interval == 0:
                plot_loss_avg = train_plot_loss_total / self.Plot_interval
                plot_losses.append(plot_loss_avg)
                train_plot_loss_total = 0

        # Plot loss figure
        plot_loss(plot_losses, Fig_name=save_road+'Loss_'+self.cell_name+'_L'+str(self.Num_layers) +
                  '_H'+str(self.Hidden_Size)+'_I'+str(self.Num_iters)+'_'+self.Optim_method)
        print('\n------------------------------------------------')
        print('LSTM Model finished fitting')
        print('------------------------------------------------')

        return self

    def predict(self, input):
        y_pred = self._predict(input)
        return y_pred

    def _predict(self,input):
        input = input.cuda()
        predict_h_state = self.initHidden(input)
        predict_c_state = self.initHidden(input)
        y_pred, predict_h_state, predict_c_state = self.forward(input, predict_h_state, predict_c_state)
        y_pred = y_pred.cpu().data.numpy()
        return y_pred


# ---------------
# MLP inherit base model
# ---------------
class mlpModel(BaseModel):
    '''
    MLP inherit base model, input: shape[batch,input_dim]
    '''
    def __init__(self, input_dim, hidden_size, output_dim, num_layers, cell, num_iters, optim_method='SGD', learning_rate=0.1, print_interval=50, plot_interval=1, view_interval=1):
        super(mlpModel, self).__init__(
            input_dim, hidden_size, output_dim, num_layers, cell)
        self.cell_name = cell
        self.Print_interval = print_interval
        self.Plot_interval = plot_interval
        self.Num_iters = num_iters
        self.Optim_method = optim_method
        self.Learn_rate = learning_rate
    
    def forward(self, input):
        # input: shape[batch,input_dim]
        mlp_output = self.Cell(input)
        # FC_Outputs = []  # save all predictions
        mlp_output=F.relu(mlp_output)
        mlp_output=self.fc(mlp_output)
        # output: shape[batch,output_dim]

        return mlp_output
    
    def fit(self, input, target, save_road='./Results/'):
        print('================================================')
        # print(self.Cell)
        print(self.cell_name+'_L'+str(self.Num_layers) + '_H' +
              str(self.Hidden_Size)+'_I'+str(self.Num_iters)+'_'+self.Optim_method)
        print('================================================\n')
        input = input.cuda()
        target = target.cuda()
        criterion = nn.MSELoss()
        if self.Optim_method == 'SGD':
            optimizer = optim.SGD(
                self.parameters(), lr=self.Learn_rate, momentum=0)
        if self.Optim_method == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.Learn_rate)

        # Initialize timer
        time_tr_start = time.time()

        plot_losses = []
        train_print_loss_total = 0  # Reset every print_every
        train_plot_loss_total = 0  # Reset every plot_every

        # plt.figure(1,figsize=(30,5))# continuously plot
        # plt.ion() # continuously plot

        # input_size=input.size(0)# continuously plot
        # time_period=np.arange(input_size)# continuously plot

        # begin to train
        for iter in range(1, self.Num_iters + 1):
            # input: shape[batch,input_dim]
            # prediction: shape[batch,output_dim=1]
            prediction = self.forward(input)
            loss = criterion(prediction, target)
            training_rmse = np.sqrt(loss.item())
            train_plot_loss_total += training_rmse
            train_print_loss_total += training_rmse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # target_view=input[:,-1,:].data.numpy()# continuously plot
            # prediction_view=prediction[:,-1,:].data.numpy()
            # plt.plot(time_period,target_view.flatten(),'r-')
            # plt.plot(time_period,prediction_view.flatten(),'b-')
            # plt.draw();plt.pause(0.05)

            if iter % self.Print_interval == 0:
                print_loss_avg = train_print_loss_total / self.Print_interval
                train_print_loss_total = 0
                print('%s (%d %d%%) ' % (timeSince(time_tr_start, iter / self.Num_iters),
                                             iter, iter / self.Num_iters * 100))
                print('Training RMSE:  \t %.3e' % (print_loss_avg))
            if iter % self.Plot_interval == 0:
                plot_loss_avg = train_plot_loss_total / self.Plot_interval
                plot_losses.append(plot_loss_avg)
                train_plot_loss_total = 0

        # Plot loss figure
        plot_loss(plot_losses, Fig_name=save_road+'Loss_'+self.cell_name+'_L'+str(self.Num_layers) +
                  '_H'+str(self.Hidden_Size)+'_I'+str(self.Num_iters)+'_'+self.Optim_method)
        print('\n------------------------------------------------')
        print('MLP Model finished fitting')
        print('------------------------------------------------')

        return self

    def predict(self, input):
        y_pred = self._predict(input)
        return y_pred

    def _predict(self,input):
        input = input.cuda()
        y_pred = self.forward(input)
        y_pred = y_pred.cpu().data.numpy()
        return y_pred