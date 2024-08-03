import json
import os
import secrets
import string
from datetime import datetime

import mysql.connector as msql
from dotenv import load_dotenv
from mysql.connector import Error
from passlib.hash import sha256_crypt
from path import Path

load_dotenv()

# ***************************************************************************** #
# Parameters.
email = "johndoe@domain.ext"
organization = 'dagnino'
password = "DEMO_dagnino_1234"
# alphabet = string.ascii_letters + string.digits
# password = ''.join(secrets.choice(alphabet) for _ in range(20))
usage_deadline_utc = datetime.fromisoformat('3000-12-31T23:59:59')
# usage_deadline_utc = datetime.now() + timedelta(days=30)
pricing_type = 'PerRequest'
n_requests_max = 1000
n_secs_max = None

print(f"password = {password}")
password_hashed = sha256_crypt.hash(password)

# ***************************************************************************** #
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

# ***************************************************************************** #
conn = msql.connect(
    user="dagnino",
    password=os.getenv('MYSQL_DAGNINO_PASSWORD'),
    host="train-deploy-db.mysql.database.azure.com",
    # host='localhost',
    port=3306,
    ssl_disabled=False)

# ***************************************************************************** #
# Add users.
try:
    if conn.is_connected():
        cursor = conn.cursor()
        cursor.execute("USE TrainDeploy_API_DB;")
        record = cursor.fetchone()

        # Add user with hashed password
        cursor.execute(
            "INSERT INTO user_access ("
            "   email, organization, password, usage_deadline_utc, pricing_type, requests_count, n_requests_max, secs_count, n_secs_max"
            ") "
            "VALUES (%s, %s, %s, %s, %s, 0, %s, 0, %s);",
            (email, organization, password_hashed, usage_deadline_utc, pricing_type, n_requests_max, n_secs_max,)
        )
        conn.commit()
        print("User added")

except Error as expt:
    print("Error while connecting to MySQL")
    print(expt)
print()
