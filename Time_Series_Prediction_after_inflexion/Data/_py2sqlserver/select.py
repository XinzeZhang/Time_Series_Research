import pyodbc 
import numpy as np
import os
# Some other example server values are
# server = 'localhost\sqlexpress' # for a named instance
# server = 'myserver,port' # to specify an alternate port
server = '127.0.0.1' 
database = 'db_load_time_series' 
username = 'sa' 
password = '1231' 
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

meters_id= 1002
cursor.execute("SELECT meters_date, meters_time, load_data from table_load_hour_particulars where meters_id = "+str(meters_id)+";") 
rows = cursor.fetchall()
meters_date_list = []
meters_time_list = []
load_data_list =[]

rows_count= len(rows)
for row in rows:
    meters_date_list.append(row.meters_date)
    meters_time_list.append(row.meters_time)
    load_data_list.append(row.load_data)
data_count=len(load_data_list)

if rows_count==data_count:
    print("select success!")

dirs = './Data/Residential_Load/'
if not os.path.exists(dirs):
    os.mkdir(dirs)
np.savez(dirs+"meters_id_"+str(meters_id)+"_hour.npz",meters_date_list,meters_time_list,load_data_list)
print("save success!")

