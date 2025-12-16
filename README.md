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
3. Create the MinIO bucket (required once):
   ```shell
   docker exec -it mlflow python create_bucket.py
   ```
4. Run training:
   ```shell
   docker exec mlflow python -m training.main
   ```
5. After train we should have the model saved in the bucket,
   to make sure you can list stored artifacts:
   ```shell
    docker exec -it minio mc ls local
   ```
   ... if authentication is necessary run:
   ```shell
   docker exec -it minio mc alias set local http://localhost:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD
   ```
7. Next we can make visualizations of the metrics collected from training and processing logs with Grafana:
    - Open http://localhost:3000 in browser,
    - Default credentials: admin/admin,
    - Add Prometheus connection like 'Connection → Data sources',
    - Now create a dashboard under 'Dashboards', similarly shown in the following image:

    ![Grafana sample](https://github.com/bencetaro/mlops_sentinel/blob/main/images/sample_grafana.png)

8. Run model inference:
    In order to run inference correctly, the input folder (./mlops_sentinel/inference/input_tci) must contain a Sentinel 2 TCI file. The script will do the rest with preprocessing, predicting and postprocessing the output. It can be initiated like:
   ```shell
   docker exec mlflow python -m inference.UNet_inference
   ```
   Notably, this is a registry-based inference pipeline, meaning that inference always loads the currently promoted 'Production' model from the MLflow model registry, allowing safe replacement via retraining and re-staging without code changes.

## ✅ What this project demonstrates:

✔ Reproducible training with Docker

✔ Experiment tracking with MLflow

✔ S3-backed artifact storage

✔ Automatic model registration & staging

✔ Metrics monitoring with Prometheus

✔ Visualization with Grafana

