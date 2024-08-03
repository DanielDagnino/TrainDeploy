import os

import mysql.connector as msql
from dotenv import load_dotenv
from mysql.connector import Error
from passlib.hash import sha256_crypt

load_dotenv()

# ***************************************************************************** #
conn = msql.connect(
    user="dagnino",
    password=os.getenv('MYSQL_DAGNINO_PASSWORD'),
    host="train-deploy-db.mysql.database.azure.com",
    port=3306,
    ssl_disabled=False)

# conn = msql.connect(
#     host='localhost',
#     user='dagnino',
#     password=os.getenv('MYSQL_DAGNINO_PASSWORD'))

# ***************************************************************************** #
# Parameters.
email = "johndoe@domain.ext"
organization = 'myself'
password = "DEMO_dagnino_1234"
print(f"password = {password}")
password_hashed = sha256_crypt.hash(password)

# ***************************************************************************** #
# Create db if it does not exist.
try:
    if conn.is_connected():
        cursor = conn.cursor()
        cursor.execute(
            "CREATE DATABASE IF NOT EXISTS TrainDeploy_API_DB;"
        )
        print("TrainDeploy_API_DB database is created or already exists")
except Error as expt:
    print("1 - Error while connecting to MySQL")
    print(expt)
print()

# ***************************************************************************** #
# Create table if not exists.
try:
    if conn.is_connected():
        cursor = conn.cursor()
        cursor.execute("USE TrainDeploy_API_DB;")
        record = cursor.fetchone()

        # # In case the database is corrupted and we need to recreate after drop the entire database:
        # # Drop table.
        # cursor.execute(f"DROP TABLE IF EXISTS user_access;")

        # Create table.
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS user_access ("
            "   id INT AUTO_INCREMENT PRIMARY KEY,"
            "   email VARCHAR(50) UNIQUE NOT NULL,"
            "   organization VARCHAR(50) NOT NULL,"
            "   password VARCHAR(128) NOT NULL,"
            "   usage_deadline_utc DATETIME,"
            "   pricing_type ENUM('PerSec', 'PerRequest', 'NoLimit') NOT NULL,"
            "   requests_count INT,"
            "   n_requests_max INT,"
            "   secs_count FLOAT,"
            "   n_secs_max INT"
            ");"
        )
        print("user_access table created or exists")

except Error as expt:
    print("2 - Error while connecting to MySQL")
    print(expt)
print()

# ***************************************************************************** #
# Add users.
try:
    if conn.is_connected():
        cursor = conn.cursor()
        cursor.execute("USE TrainDeploy_API_DB;")
        record = cursor.fetchone()

        # Add user
        cursor.execute(
            "INSERT INTO user_access ("
            "   email, organization, password, usage_deadline_utc, pricing_type, requests_count, n_requests_max, secs_count, n_secs_max) "
            "VALUES (%s, %s, %s, NULL, %s, 0, NULL, 0, NULL);",
            (email, organization, password_hashed, 'PerSec')
        )
        conn.commit()
        print("User added")

except Error as expt:
    print("3 - Error while connecting to MySQL")
    print(expt)
print()
