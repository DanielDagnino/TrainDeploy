# Console: access to MySQL
# mysql -h localhost -u dagnino -p

# Create user + grant access
CREATE USER 'dagnino'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON *.* TO 'dagnino'@'localhost' WITH GRANT OPTION;
FLUSH PRIVILEGES;

# Change password
ALTER USER 'dagnino'@'localhost' IDENTIFIED BY 'new_password';
ALTER USER 'root'@'localhost' IDENTIFIED BY 'new_password';

# Create db
CREATE DATABASE TRAINDEPLOY_API_DB;
SHOW DATABASES;
