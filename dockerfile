#Python base image
FROM python:3.11-slim

# Root to the directory for the DOCKER
WORKDIR /app

# Install packages before copying the code so this layer stays cached
# and only reruns when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (see .dockerignore for what is left out)
COPY . .

CMD ["gunicorn", "src.main:app", "--workers", "3", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
