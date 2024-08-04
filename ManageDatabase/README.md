# Example to create and configure a MySQL DB in Azure

## Create the DB
Got to `Azure Database for MySQL servers` and create the MySQL DB. In this example I create the following DB:
```
hostname=train-deploy-db.mysql.database.azure.com
username=dagnino
password=...
```

## Set permissions
### Access to the DB
Add the desired IPs in the `Networking>Firewall rules` to be capable to connect.

Test: 
```bash
mysql -h train-deploy-db.mysql.database.azure.com -u dagnino -p
```

### SSL certifcate to connect
Download one certificate from (link from `Networking`) to create connections:
https://learn.microsoft.com/en-gb/azure/postgresql/flexible-server/concepts-networking-ssl-tls#downloading-root-ca-certificates-and-updating-application-clients-in-certificate-pinning-scenarios

Convert to PEM format (example DigiCertGlobalRootCA.crt):
```bash
openssl x509 -inform DER -in DigiCertGlobalRootCA.crt -out DigiCertGlobalRootCA.pem -outform PEM
```

Test: 
```python
DATABASE_URL = 'mysql+pymysql://' \
               f'dagnino:{MYSQL_DAGNINO_PASSWORD}@{DB_HOST}:3306/TrainDeploy_API_DB?ssl_ca={PATH_SSL_CA_CERTIFICATE}'

_engine = create_engine(
                DATABASE_URL,
                ...
            )
```

## Manipulate the DB
In this directory I added some useful scripts to manipulate the DB, namely create the main table, and add/rmv users.
