#Python base image
FROM python:3.11-slim

# Root to the directory for the DOCKER
WORKDIR /src

COPY requirements.txt .

# Copy the entire project into the image
COPY . .

# Build docker image from MLflow model
RUN pip install --no-cache-dir -r requirementss.txt

CMD ["gunicorn", "main:app", "--workers", "3", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]