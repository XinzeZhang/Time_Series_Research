import numpy as np
import os


dirs = "./Data/Crude_Oil_Price/ED_Var/"

dir_list = os.listdir(dirs)

for folder in dir_list:
    sh = './train_sh/train_'+folder+'.sh'
    sh = sh.replace('WTI_', '', -1)
    with open(sh, 'w') as f:
        print('cd .. \n', file=f)
        for size in [256, 512, 1024, 2048, 4096]:
                print('\
python3 rnn_train.py --cell RNN --hidden_size ' + str(size) + ' --dir ' + folder + ' --num_iters 10000 \n\
python3 rnn_train.py --cell GRU --hidden_size ' + str(size) + ' --dir ' + folder + ' --num_iters 10000 \n\
python3 rnn_train.py --cell LSTM --hidden_size ' + str(size) + ' --dir ' + folder + ' --num_iters 10000 \n\
python3 mlp_train.py --cell Linear --hidden_size ' + str(size) + ' --dir ' + folder + ' --num_iters 10000 \n\
                    ', file=f)
        print('python3 svr_train.py' + ' --dir ' + folder + ' \n', file=f)


# data = []
# data.append([1,2])
# data.append([1,2,3])
# data=np.array(data)
    print(sh)
