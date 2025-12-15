# MLOps pipeline using multiple services (MLflow, S3, Prometheus, Grafana)
### Full ML life-cycle (Training → Registry → Inference)

This repository demonstrates a complete MLOps workflow for training, tracking, registering, and deploying a Pytorch segmentation model.

**Note:** This project currently supports cpu only training, and the experiment shown in the follow up is only a 1 epoch train, but the goal was here to demonstrate the learning process of my MLOps skills.

## Featuring:
- MLflow — experiment tracking & model registry
- MinIO (S3-compatible) — artifact & model storage
- Prometheus — training metrics collection
- Grafana — metrics visualization
- Docker Compose — reproducible orchestration

## 📁 Project Structure:
    .
    ├── docker-compose.yml
    ├── docker/
    │   └── Dockerfile
    ├── training/
    │   ├── main.py
    │   ├── trainer.py
    │   ├── model.py
    │   ├── dataset.py
    │   └── etc...
    ├── inference/
    │   ├── UNet_inference.py
    │   ├── inf_config.yml
    │   └── etc...
    ├── monitoring/
    │   ├── prometheus.yml
    │   └── grafana/
    ├── create_bucket.py
    ├── requirements.txt
    ├── Makefile
    ├── .env
    └── README.md

## 🔐 Environment variables:
Create a similar .env file in the project root:

    MLFLOW_TRACKING_URI=http://mlflow:5000
    MLFLOW_S3_ENDPOINT_URL=http://minio:9000
    AWS_ACCESS_KEY_ID=minio_admin
    AWS_SECRET_ACCESS_KEY=minio_password
    AWS_DEFAULT_REGION=us-east-1
    AWS_BUCKET_NAME=mlflow-artifacts

## 🚀 Setup guide:
(Most of the bash/docker commands can be easily understood from the included Makefile)

1. Build and start services from the compose file:
   ```shell
   docker compose build --no-cache
   docker compose up -d
   ```
2. Check out certain services:
   For mlflow write...
   ```shell
   docker compose logs mlflow -f --tail=30
   ```  
4. Create the MinIO bucket (required once):
   ```shell
   docker exec -it mlflow python create_bucket.py
   ```
5. Run training:
   ```shell
   docker exec mlflow python -m training.main
   ```
6. After train we should have the model saved in the bucket,
   to make sure you can:
   ... continue...
   
   ```shell
   docker exec -it mlflow python create_bucket.py
   ```
8. fsaf
   ```shell
   docker exec -it mlflow python create_bucket.py
   ```
   










