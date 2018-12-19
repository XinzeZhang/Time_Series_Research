import numpy as np

dirs="./Data/Crude_Oil_Price/WTI_1_53"
temp=np.load(dirs+"/trainSet.npz")
data=temp["arr_0"]
print(data)