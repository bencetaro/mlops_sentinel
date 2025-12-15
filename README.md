# MLOps pipeline using multiple services (MLflow, S3, Prometheus, Grafana)
### Full ml lifecycle (Training → Registry → Inference)

This repository demonstrates a complete MLOps workflow for training, tracking, registering, and deploying a Pytorch segmentation model.

## Featuring:
- MLflow — experiment tracking & model registry
- MinIO (S3-compatible) — artifact & model storage
- Prometheus — training metrics collection
- Grafana — metrics visualization
- Docker Compose — reproducible orchestration

# 📁 Project Structure:
    .
    ├── docker-compose.yml
    ├── docker/
    │   └── Dockerfile
    ├── training/
    │   ├── main.py
    │   ├── trainer.py
    │   ├── model.py
    │   └── dataset.py
    ├── inference/
    │   ├── UNet_inference.py
    │   └── inf_config.yml
    ├── monitoring/
    │   ├── prometheus.yml
    │   └── grafana/
    ├── create_bucket.py
    ├── .env
    └── README.md

# 🔐 Environment variables:
Create a similar .env file in the project root:

    MLFLOW_TRACKING_URI=http://mlflow:5000
    MLFLOW_S3_ENDPOINT_URL=http://minio:9000
    AWS_ACCESS_KEY_ID=minio_admin
    AWS_SECRET_ACCESS_KEY=minio_password
    AWS_DEFAULT_REGION=us-east-1
    AWS_BUCKET_NAME=mlflow-artifacts

# 🚀 Setup guide:
1. 




