FROM ubuntu:22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.10 python3-pip python3.10-venv \
    ffmpeg libasound2-dev \
    git nano

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN python3 -m venv /venv && /venv/bin/python3 -m pip install --upgrade pip

WORKDIR /home

COPY apis /home/apis
COPY .gitmodules /home
COPY requirements.txt /home
COPY DigiCertGlobalRootCA.pem /home

RUN /venv/bin/python3 -m pip install --no-cache-dir --upgrade -r requirements.txt

#RUN git init && mkdir submodules && git submodule init && git submodule update --init --recursive
RUN mkdir submodules && cd submodules && git submodule add --force https://github.com/microsoft/unilm.git

RUN find . -name "*.egg-info" -exec rm -rf {} + && \
    find . -name "__pycache__" -exec rm -rf {} + && \
    find . -name "*.pyc" -exec rm -rf {} + && \
    find . -name "audio_search/*tmp*" -exec rm -rf {} + && \
    find . -name "submodules/*tmp*" -exec rm -rf {} + && \
    find . -name "*.png" -exec rm -rf {} + && \
    find . -name "*.jpeg" -exec rm -rf {} + && \
    find . -name "*.jpg" -exec rm -rf {} +

EXPOSE 8000

ENV PYTHONPATH=".:submodules/unilm/beats"

#CMD ["uvicorn", "apis.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["/venv/bin/python3", "apis/main_api_db.py"]
