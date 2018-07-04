import pyodbc 
import numpy as np
import os
from typing import List
# Some other example server values are
# server = 'localhost\sqlexpress' # for a named instance
# server = 'myserver,port' # to specify an alternate port
server = '127.0.0.1' 
database = 'db_load_time_series' 
username = 'sa' 
password = '1231' 
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()


cursor.execute("select id from table_meter_types where meters_type = 1")
rows = cursor.fetchall()
meter_id_list=[]
for row in rows:
    meter_id_list.append(row.id)
print(len(meter_id_list))

# load_lists = []
load_lists: List[list] = []
indexs = list(range(len(meter_id_list)))
for  index, meters_id in zip(indexs,meter_id_list):
    load_lists.append([])
    sql="select load_data from table_load_hour_particulars where meters_id = "+str(meters_id)+";"
    print(sql)
    cursor.execute(sql) 
    rows = cursor.fetchall()
    rows_count= len(rows)
    for row in rows:
        load_lists[index].append(row.load_data)
    print(len(load_lists[index]))
print("select success!")

load_array = np.array(load_lists)
load_total = load_array.sum(0)
print(load_total.shape)
# data_count=len(load_data_list)

# if rows_count==data_count:
#     print("select success!")

dirs = './Data/Residential_Load/'
if not os.path.exists(dirs):
    os.mkdir(dirs)
np.savez(dirs+"Residential_Load_hour.npz",load_total)
print("save success!")

