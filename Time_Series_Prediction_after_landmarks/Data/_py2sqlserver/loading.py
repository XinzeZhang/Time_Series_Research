import pyodbc 
import pandas as pd


server = 'localhost' 
database = 'db_load_time_series' 
username = 'sa' 
password = '1231' 
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

# the file need to be bulk inserted should be in the same hard disk with sql server", otherwise it's diffcult to solve the error prompted
sql = "Bulk insert table_meter_types \
From 'C:/load_separately_ordered/type.txt' \
With\
(   fieldterminator='\t',\
    rowterminator='\n'\
) "
cursor.execute(sql)
cnxn.commit()
print("====================================================================")
print(sql)

# the file need to be bulk inserted should be in the same hard disk with sql server", otherwise it's diffcult to solve the error prompted
type = pd.read_table('C:/load_separately_ordered/type.txt', header=None, index_col=0)[1]
for i in type.index:
    if type[i]==1 :
        sql = "Bulk insert table_load_particulars \
        From 'C:/load_separately_ordered/" + str(i) + ".txt' \
        With\
        (   fieldterminator='\t',\
            rowterminator='\n'\
        ) "
        cursor.execute(sql)
        cnxn.commit()
        print("====================================================================")
        print(sql)

cursor.close()
cnxn.close()
