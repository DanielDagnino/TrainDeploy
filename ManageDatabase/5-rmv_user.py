from datetime import datetime
import os
import mysql.connector as msql
from mysql.connector import Error
from passlib.hash import sha256_crypt
from dotenv import load_dotenv

load_dotenv()

# Set connection.
conn = msql.connect(
    user="dagnino",
    password=os.getenv('MYSQL_DAGNINO_PASSWORD'),
    host="train-deploy-db.mysql.database.azure.com",
    port=3306,
    # database="{your_database}",
    # ssl_ca="{ca-cert filename}",
    ssl_disabled=False)

# Rmv users.
try:
    if conn.is_connected():
        cursor = conn.cursor()
        cursor.execute("USE TrainDeploy_API_DB;")
        record = cursor.fetchone()

        # Parameters.
        organization = 'yyy'

        # Remove user with hashed password
        cursor.execute("DELETE FROM user_access WHERE organization = %s;", (organization,))
        conn.commit()
        print("Remove organization")

except Error as expt:
    print("3 - Error while connecting to MySQL")
    print(expt)
print()
