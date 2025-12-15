FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install --default-timeout=200 --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

EXPOSE 5000

# CMD ["mlflow", "server", "--backend-store-uri", "file:/app/mlruns", "--default-artifact-root", "file:/app/mlruns", "--host", "0.0.0.0", "--port", "5000"]
CMD ["mlflow", "server", "--backend-store-uri", "file:/app/mlruns", "--default-artifact-root", "s3://mlflow-artifacts", "--host", "0.0.0.0", "--port", "5000"]

