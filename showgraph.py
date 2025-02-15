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

df.head(10)
df.info()
df['CommonCold'].value_counts()

# Count occurrences of each category
counts = df['CommonCold'].value_counts()
percentages = (counts / counts.sum()) * 100  # Convert to percentages

# Create the figure
plt.figure(figsize=(8, 6))  # Set figure size
ax = counts.plot(kind='barh', color=sns.color_palette('Dark2'))

# Remove top and right spines
ax.spines[['top', 'right']].set_visible(False)

# Set labels and title
plt.xlabel("Count")
plt.ylabel("CommonCold Category")
plt.title("Distribution of CommonCold Cases")

# Annotate bars with count and percentage
for index, value in enumerate(counts):
    plt.text(value + 1, index, f"{value} ({percentages.iloc[index]:.1f}%)", va='center', fontsize=12)

# Show the plot
plt.show()
