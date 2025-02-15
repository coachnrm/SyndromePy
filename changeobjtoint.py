import mysql.connector
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn as sk 
import seaborn as sns

# Connect to the MySQL database
conn = mysql.connector.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123456',
    database='test2',
)

# Write the SQL query
query = "SELECT * FROM ColdSyndromeInsert"

# Use pandas to execute the query and fetch the data into a DataFrame
df = pd.read_sql(query, conn)

# Display the DataFrame
# print(df)

# Close the connection
conn.close()

# หากต้องการให้แสดงข้อมูลให้เขียน print(<>) เช่น print(df.head(10))

# ต้องการ check ว่า property ไหนเป็น object
isObject = [name for name in df.columns if df[name].dtype == 'object']
print(isObject)

# การแปลง object ให้เป็น int64
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for i in list(isObject):
    df[i] = le.fit_transform(df[i])
for x in isObject:
    print(x," = ",df[x].unique())
