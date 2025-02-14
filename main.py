import mysql.connector
import pandas as pd

# Connect to the MySQL database
conn = mysql.connector.connect(
    host='localhost',
    port=3307,
    user='root',
    password='123456',
    database='test',
)

# Write the SQL query
query = "SELECT * FROM SyndromeInsert"

# Use pandas to execute the query and fetch the data into a DataFrame
df = pd.read_sql(query, conn)

# Display the DataFrame
print(df)

# Close the connection
conn.close()
