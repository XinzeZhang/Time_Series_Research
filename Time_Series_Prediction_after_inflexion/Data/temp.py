import numpy as np

data = np.loadtxt("./Data/Crude_Oil_Price/WTI.csv")
data = data.tolist()
np.savez("./Data/Crude_Oil_Price/WTI.npz",data)