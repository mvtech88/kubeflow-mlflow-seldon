FROM python:3.10-slim
WORKDIR /app
RUN apt-get update \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Port for GRPC
EXPOSE 5000
# Port for REST
EXPOSE 9000

# Define environment variables
ENV MODEL_NAME Inference
ENV SERVICE_TYPE MODEL
ENV MODEL_URI "s3://modeloutput/experiments/1/b3b05490b57b4a5ea7373a77e8b09dcd/artifacts/CNN-TinyVG" 
# Changing folder to default user
RUN chown -R 8888 /app

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE
