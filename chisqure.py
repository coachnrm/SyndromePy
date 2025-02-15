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

# Compute chi-square scores
chi_scores = chi2(X, y)
p_values = pd.Series(chi_scores[1], index=X.columns)
p_values.sort_values(ascending=True, inplace=True)

# Display results in a new Matplotlib window
plt.figure(figsize=(8, 4))  # Set figure size
plt.axis("off")  # Hide axes

# Create a table with sorted p-values
table = plt.table(cellText=p_values.reset_index().values,
                  colLabels=["Feature", "P-Value"],
                  cellLoc="center",
                  loc="center")

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Adjust table size

plt.title("Chi-Square Test P-Values")

# Show the table in a new window
plt.show()