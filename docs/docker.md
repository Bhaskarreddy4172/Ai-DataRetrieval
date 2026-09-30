# Docker & Multi-Container Deployment

## Overview
The application can be deployed using Docker and Docker Compose, spinning up both the FastAPI application and the local Ollama LLM service with shared networks and persistent volumes.

## Docker Compose
```bash
# Build and start all services
docker-compose up -d --build

# Inspect logs
docker-compose logs -f chatbot-app

# Pull model in Ollama container
docker exec -it dataset-ollama ollama pull llama3.1:8b

# Shut down services
docker-compose down
```

## Standalone Docker Build
```bash
docker build -t dataset-chatbot:latest .
docker run -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 dataset-chatbot:latest
```

