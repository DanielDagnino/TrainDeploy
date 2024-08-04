# API

## Introduction
This folder contains the minimal representation of the codes needs to deploy the model, run a server to make predictions, and to restrict the API usage to specific users and usage limits.

## Run the API locally
### Python Environment
Install environment
```bash
    cd <PATH_TO_TrainDeploy/api>
    mkvirtualenv api -p python3.10
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    export PYTHONPATH=.:submodules/unilm/beats
```

### Submodules
Install submodules
```bash
git submodule init
git submodule update --init --recursive
```

Note: Clean submodule cache:
```bash
git config --file .gitmodules --get-regexp path | awk '{ print $2 }'
git rm --cached <PATH_SUBMODULE>
git submodule add --force <HTTP_LINK>
```

### Make GPU available in the Docker container
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.10.0/install-guide.html
```bash
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
```

### Build/Run/Test container
Build
```bash
docker build -t train_deploy_api:latest .
docker tag train_deploy_api:latest train_deploy_api:v1.0.0
```

Run Test
```bash
docker run -it -p 8000:8000 -e DB_HOST=train-deploy-db.mysql.database.azure.com train_deploy_api:v1.0.0
docker run -it -p 8000:8000 -e DB_HOST=train-deploy-db.mysql.database.azure.com --entrypoint bash train_deploy_api:v1.0.0
docker run -it -p 8000:8000 -e DB_HOST=train-deploy-db.mysql.database.azure.com --gpus '"device=0"' train_deploy_api:v1.0.0
```

Debugging: Revise Docker running
```bash
docker exec -it <CONTAINER_ID> bash
```

Test
```bash
./tests/requester.py
```
