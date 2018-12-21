cd .. 
 python3 rnn_train.py --cell RNN --hidden_size 256 --dir WTI_1_53 --num_iters 10000 
 python3 rnn_train.py --cell RNN --hidden_size 512 --dir WTI_1_53 --num_iters 10000 
 python3 rnn_train.py --cell RNN --hidden_size 1024 --dir WTI_1_53 --num_iters 10000 
 python3 rnn_train.py --cell RNN --hidden_size 4096 --dir WTI_1_53 --num_iters 10000 
 python3 rnn_train.py --cell GRU --hidden_size 256 --dir WTI_1_53 --num_iters 10000 
 python3 rnn_train.py --cell GRU --hidden_size 512 --dir WTI_1_53 --num_iters 10000 
 python3 rnn_train.py --cell GRU --hidden_size 1024 --dir WTI_1_53 --num_iters 10000 
 python3 rnn_train.py --cell GRU --hidden_size 4096 --dir WTI_1_53 --num_iters 10000 
 python3 lstm_train.py --cell LSTM --hidden_size 256 --dir WTI_1_53 --num_iters 10000 
 python3 lstm_train.py --cell LSTM --hidden_size 512 --dir WTI_1_53 --num_iters 10000 
 python3 lstm_train.py --cell LSTM --hidden_size 1024 --dir WTI_1_53 --num_iters 10000 
 python3 lstm_train.py --cell LSTM --hidden_size 4096 --dir WTI_1_53 --num_iters 10000
