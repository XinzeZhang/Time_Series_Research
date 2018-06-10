import numpy as np

range_data = [1, 2, 3, 4, 5, 6,7, 8]
range_date=["a","b","c","d","e","f","g","h"]
print(range_data[1::2])
print(range_data[::2])

hour_load_data_array_1=np.array(range_data[1::2])
hour_load_data_array_2=np.array(range_data[::2])
hour_load_data_array=hour_load_data_array_1+hour_load_data_array_2
hour_load_data_array=hour_load_data_array.tolist()

print(hour_load_data_array_1+hour_load_data_array_2)

for data, date,load in zip(range_data, range_date,hour_load_data_array):
    print(data,date,load)