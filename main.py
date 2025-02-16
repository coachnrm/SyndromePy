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
X = df[['Head','Nose','Neck','Fever']]
y = df['CommonCold']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

modelRF = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
modelRF.fit(X_train, y_train)

y_pred = modelRF.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))

import pickle
filename = 'common-cold.sav'
pickle.dump(modelRF,open(filename, 'wb'))

load_model = pickle.load(open(filename,'rb'))
# print(load_model.predict([[1, 1, 1, 1]]))


# Define column names based on the training data
feature_names = ['Head', 'Nose', 'Neck', 'Fever']

# Convert input to a DataFrame
input_data = pd.DataFrame([[1, 1, 1, 0]], columns=feature_names)

# Make prediction
print(load_model.predict(input_data))










