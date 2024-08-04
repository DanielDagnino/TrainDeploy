from datetime import datetime
import json
import os
import secrets
import string

import mysql.connector as msql
from dotenv import load_dotenv
from mysql.connector import Error
from passlib.hash import sha256_crypt
from path import Path

load_dotenv()

# Parameters.
email = os.getenv('DEMO_USER_EMAIL')
password = os.getenv('DEMO_USER_PASSWORD')
organization = 'dagnino'
usage_deadline_utc = None
pricing_type = 'PerSec'
n_requests_max = None
n_secs_max = None

if password is not None:
    print(f"password = {password}")
    password_hashed = sha256_crypt.hash(password)
else:
    password_hashed = None

# Save user data in a JSON.
data = json.load(open("users.json")) if Path("users.json").exists() else dict()
data[email] = {
    "email": email,
    "password": password,
    "organization": organization,
    "usage_deadline_utc": usage_deadline_utc.isoformat() if usage_deadline_utc is not None else None,
    "pricing_type": pricing_type,
    "n_requests_max": n_requests_max,
    "n_secs_max": n_secs_max,
}
json.dump(data, open("users.json", "w"), indent=4)

# Set connection.
conn = msql.connect(
    user="dagnino",
    password=os.getenv('MYSQL_DAGNINO_PASSWORD'),
    host="train-deploy-db.mysql.database.azure.com",
    port=3306,
    ssl_disabled=False)

# Update users.
try:
    if conn.is_connected():
        cursor = conn.cursor()
        cursor.execute("USE TrainDeploy_API_DB;")
        record = cursor.fetchone()
        if n_secs_max is not None:
            cursor.execute("UPDATE user_access SET n_secs_max = %s WHERE organization = %s;",
                           (n_secs_max, organization))
        if n_requests_max is not None:
            cursor.execute("UPDATE user_access SET n_requests_max = %s WHERE organization = %s;",
                           (n_requests_max, organization))
        if usage_deadline_utc is not None:
            cursor.execute("UPDATE user_access SET usage_deadline_utc = %s WHERE organization = %s;",
                           (usage_deadline_utc, organization))
        if password_hashed is not None:
            cursor.execute("UPDATE user_access SET password = %s WHERE email = %s;", (password_hashed, email))

        conn.commit()
        print("User added")

except Error as expt:
    print("3 - Error while connecting to MySQL")
    print(expt)
print()
