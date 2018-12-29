import pyodbc 
import numpy as np
import os
import sys
import pandas as pd
# Some other example server values are
# server = 'localhost\sqlexpress' # for a named instance
# server = 'myserver,port' # to specify an alternate port
server = '127.0.0.1' 
database = 'db_load_time_series' 
username = 'sa' 
password = '1231' 
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

type = pd.read_table('C:/load_separately_ordered/type.txt', header=None, index_col=0)[1]
for meters_id in type.index:
    if type[meters_id]==1 :

        cursor.execute("SELECT meters_id, meters_date, meters_time, load_data from table_load_halfhour_particulars where meters_id = "+str(meters_id)+";") 
        rows = cursor.fetchall()
        meters_id_list = []
        meters_date_list = []
        meters_time_list = []
        load_data_list =[]

        rows_count= len(rows)
        for row in rows:
            meters_id_list.append(row.meters_id)
            meters_date_list.append(row.meters_date)
            meters_time_list.append(row.meters_time)
            load_data_list.append(row.load_data)
        data_count=len(load_data_list)

        if rows_count==data_count:
            print("select success! Total length is: "+str(data_count))

        hour_meters_id_list=meters_id_list[::2]
        hour_meters_date_list=meters_date_list[::2]

        hour_meters_time_list=meters_time_list[1::2]
        hour_meters_time_array=np.array(hour_meters_time_list)/2
        hour_meters_time_list=hour_meters_time_array.tolist()

        hour_load_data_array_1=np.array(load_data_list[1::2])
        hour_load_data_array_2=np.array(load_data_list[::2])

        hour_load_data_array=hour_load_data_array_1+hour_load_data_array_2
        hour_load_data_list=hour_load_data_array.tolist()

        with open("C:/load_separately_ordered/"+str(meters_id)+"_hour.txt", "a+") as f:
            for meter_id, date, time , load in zip(hour_meters_id_list,hour_meters_date_list,hour_meters_time_list,hour_load_data_list):
                print("%s\t%s\t%i\t%.3f"% (meter_id,date,time,load),file=f)
        print("Saved "+str(meters_id)+" hour_load!")

        sql = "Bulk insert table_load_hour_particulars \
        From 'C:/load_separately_ordered/" + str(meters_id) + "_hour.txt' \
        With\
        (   fieldterminator='\t',\
            rowterminator='\n'\
        ) "
        cursor.execute(sql)
        cnxn.commit()
        print("bulk inserted hour_load "+str(meters_id))
        print("====================================================================")
        # print(sql)

print("finish!")