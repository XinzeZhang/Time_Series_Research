from __future__ import print_function

from data_process._data_process import create_dataset, plot_forecasting_result
import os
import argparse
from typing import List
from _definition import pivot_k_window, MDPP
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from sklearn.svm import SVR

import numpy as np
from numpy import concatenate, atleast_2d

import matplotlib
matplotlib.use('agg')  # avoiding Invalid DISPLAY variable
# matplotlib.use('Agg')


# Training settings
# ==============================================================================
parser = argparse.ArgumentParser(
    description='PyTorch Time Series Forecasting after Landmarks')
parser.add_argument('--kernel', type=str, default='rbf', metavar='S',
                    help='cell types for training (default: rbf)')
parser.add_argument('--C', type=float, default=1e3, metavar='F',
                    help='Penalty parameter C of the error term (default: 1e3)')
parser.add_argument('--gamma', type=float, default=0.1, metavar='F',
                    help='Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’ (default: 0.1)')
parser.add_argument('--dir', type=str, default="WTI_1_53", metavar='S',
                    help='dir of training data')

if __name__ == '__main__':
    args = parser.parse_args()

    input_dir = "./Data/Crude_Oil_Price/ED_12/"+args.dir + "/"
    result_dir = "./Results/COP/ED_12/"+args.dir + "/"
    print('\n------------------------------------------------')
    print('Loading Data: '+input_dir)
    print('------------------------------------------------')
    raw = np.load(input_dir+"/rawSet.npz")
    raw = raw["arr_0"]
    raw_T = raw.shape[0]
    raw_section = [*range(raw_T)]
    raw_values = raw.tolist()

    data = np.load(input_dir+"/dataSet.npz")
    train, test = data["arr_0"], data["arr_1"]
    train_N, train_T = train.shape[0], train.shape[1]
    test_N, test_T = test.shape[0], test.shape[1]

    idx = np.load(input_dir+"/idxSet.npz")
    train_idx, test_idx = idx["arr_0"], idx["arr_1"]

    #  shape of training data should be (batch,input-dim)
    train_input = atleast_2d(train[:, :-1])[:, :]
    # shape of target should be (batch,) for SVR
    train_target = train[:, -1]
    # --

    train_section = train_idx[:, -1].flatten().tolist()

    test_input = atleast_2d(test[:, :-1])[:, :]
    test_target = test[:, -1]
    # --

    test_section = test_idx[:, -1].flatten().tolist()
    test_section.insert(0, train_section[-1])

    svr_demo = SVR(kernel=args.kernel, C= args.C, gamma=args.gamma)
    print('\n------------------------------------------------')
    print(svr_demo)
    print('------------------------------------------------')

    svr_demo.fit(train_input,train_target)

    # ---------
    # begin to forcast
    print('\n------------------------------------------------')
    print('Forecasting Testing Data')
    print('------------------------------------------------')

    train_pred = svr_demo.predict(train_input)
    # get test_result
    test_pred = svr_demo.predict(test_input)
    np.savez(result_dir+'Npz_'+'SVR_'+args.kernel.upper() +".npz", train_pred, test_pred)

    print('\n------------------------------------------------')
    print('Ploting Testing Data')
    print('------------------------------------------------')

    MSE_pred= mean_squared_error(test_target,test_pred)
    RMSE_pred = np.sqrt(MSE_pred)

    plot_fig_name = result_dir+'Fig_'+'SVR_'+args.kernel.upper() 

    train_pred_plot = train_pred.flatten()

    test_pred_plot = test_pred.flatten().tolist()
    test_pred_plot.insert(0, train_target.flatten()[-1])

    # ============
    plt.figure(figsize=(20, 5))
    # plt.title(
    #     'Forecasting Future Values for Time Series', fontsize=12)
    plt.title('RMSE of Prediction: %(rmse).3e' %
              {'rmse': RMSE_pred}, fontsize=10)
    plt.xlabel('Input Sequence', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.plot(raw_section, raw_values, 'k-', label='Raw Series', linewidth=1)
    # plt.plot(train_section, train_target_plot, 'c-', label='Training Target', linewidth=1)
    plt.plot(train_section, train_pred_plot, 'm-',
             label='Training Result', linewidth=1)
    # plt.plot(test_idx[:,-1], test[:,-1], 'b-.', label='Test Target', linewidth=1)
    plt.plot(test_section, test_pred_plot, 'r-.',
             label='Test Result', linewidth=1)
    plt.legend(loc='upper right')
    plt.savefig(plot_fig_name + '.png')