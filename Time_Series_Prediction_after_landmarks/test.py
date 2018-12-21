import numpy as np
import os



dirs="./Data/Crude_Oil_Price/ED_Var/"

dir_list = os.listdir(dirs)

for folder in dir_list:
    sh='./train_sh/train_'+folder+'.sh'
    sh = sh.replace('WTI','',-1)
    with open(sh,'a+') as f:
        print('cd .. \n \
python3 rnn_train.py --cell RNN --hidden_size 256 --dir ' + folder + ' --num_iters 10000 \n \
python3 rnn_train.py --cell RNN --hidden_size 512 --dir ' + folder + ' --num_iters 10000 \n \
python3 rnn_train.py --cell RNN --hidden_size 1024 --dir ' + folder + ' --num_iters 10000 \n \
python3 rnn_train.py --cell RNN --hidden_size 4096 --dir ' + folder + ' --num_iters 10000 \n \
python3 rnn_train.py --cell GRU --hidden_size 256 --dir ' + folder + ' --num_iters 10000 \n \
python3 rnn_train.py --cell GRU --hidden_size 512 --dir ' + folder + ' --num_iters 10000 \n \
python3 rnn_train.py --cell GRU --hidden_size 1024 --dir ' + folder + ' --num_iters 10000 \n \
python3 rnn_train.py --cell GRU --hidden_size 4096 --dir ' + folder + ' --num_iters 10000 \n \
python3 lstm_train.py --cell LSTM --hidden_size 256 --dir ' + folder + ' --num_iters 10000 \n \
python3 lstm_train.py --cell LSTM --hidden_size 512 --dir ' + folder + ' --num_iters 10000 \n \
python3 lstm_train.py --cell LSTM --hidden_size 1024 --dir ' + folder + ' --num_iters 10000 \n \
python3 lstm_train.py --cell LSTM --hidden_size 4096 --dir ' + folder + ' --num_iters 10000', file = f)

# data = []
# data.append([1,2])
# data.append([1,2,3])
# data=np.array(data)
    print(sh)