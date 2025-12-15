include .env
export
	
up:
	@docker compose up -d

down:
	@docker compose down -v

logs:
	@docker compose logs mlflow -f --tail=30
	@docker compose logs minio -f --tail=30

createbucket:
	@docker exec -it mlflow python create_bucket.py

miniostart:
# first check that the artifact exists: docker exec -it minio mc ls local -> should show the bucket
	@docker exec -it minio mc alias set local http://localhost:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}

train:
# before train, run: docker exec mlflow python create_bucket.py
	@docker exec -it mlflow python -m training.main

inference:
	@docker exec -it mlflow python -m inference.UNet_inference

ps:
	@docker compose ps

rebuild:
	@echo "Rebuilding all images..."
	@docker compose --env-file .env build --no-cache

restart:
	@echo "Restarting all services..."
	@docker compose restart

clean:
	@echo "Cleaning unused Docker resources..."
	@docker compose down -v --rmi all --remove-orphans
	@docker system prune -af
	@docker volume prune -af
	@docker network prune -f

urls:
	@echo ""
	@echo " Access points:"
	@echo " - MLflow UI:        http://localhost:5000"
	@echo " - Prometheus:       http://localhost:9090"
	@echo " - Grafana:          http://localhost:3000"
	@echo ""
