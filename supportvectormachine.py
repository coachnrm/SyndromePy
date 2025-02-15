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

df.head(10) # show ข้อมูล 10 แถวแรก
df.info() # show properties ของตารางทั้งหมด
df['CommonCold'].value_counts() # นับข้อมูลของ column ชื่อ CommonCold โดยแยกว่ามีจำนวนเท่าไร

from sklearn.feature_selection import chi2

# Prepare features and target variable
X = df[['Head','Neck']]
y = df['CommonCold']

from sklearn.model_selection import train_test_split
from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

modelSVM = svm.SVC(kernel='linear')
modelSVM.fit(X_train, y_train)


y_pred = modelSVM.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))