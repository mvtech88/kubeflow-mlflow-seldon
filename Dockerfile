FROM python:3.10-slim
RUN apt-get update \
    && rm -rf /var/lib/apt/lists/*
COPY . /microservice
WORKDIR /microservice
RUN pip install -r requirements.txt
