import numpy as np

dirs="./Data/Crude_Oil_Price/ED_Var/WTI_1_53"
temp=np.load(dirs+"/trainSet.npz")
data=temp["arr_0"]


# data = []
# data.append([1,2])
# data.append([1,2,3])
# data=np.array(data)
print(data)