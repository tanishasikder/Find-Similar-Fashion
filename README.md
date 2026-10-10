# 👗 Fashion Similarity Search

A cloud-deployed, image-based fashion discovery application that helps users find visually similar clothing items. Upload an image of an outfit or clothing item, and the application uses a PyTorch computer vision model to analyze its visual features and retrieve similar items from a fashion dataset.

## ✨ Overview

Finding clothes that match a particular style, color, or design can be difficult when browsing online stores. Fashion Similarity Search simplifies the process by allowing users to search using images instead of relying on text descriptions or product keywords.

The application processes an uploaded image, extracts relevant visual features, and compares them against clothing images in its dataset to identify similar items.

The project uses the **Fashionpedia dataset** for fashion image analysis and combines machine learning with asynchronous task processing and cloud deployment to deliver a scalable application.

## 🚀 Features

- **Image-Based Search:** Upload a clothing image to discover visually similar fashion items.
- **Visual Feature Extraction:** Use a PyTorch model to extract clothing characteristics for similarity comparisons.
- **Fashion Dataset Integration:** Leverage the Fashionpedia dataset for clothing imagery and fashion-related annotations.
- **Asynchronous Processing:** Use Celery and Redis to handle computationally intensive tasks outside the main API request cycle.
- **REST API:** Expose application functionality through FastAPI endpoints.
- **Containerization:** Package the application and its dependencies with Docker for consistent deployment.
- **Cloud Deployment:** Deploy the application in a cloud environment for remote access and scalable operation.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and ML development |
| PyTorch | Computer vision and visual feature extraction |
| Fashionpedia | Fashion image dataset and annotations |
| FastAPI | REST API and image-upload endpoints |
| Celery | Background task execution |
| Redis | Message broker and task coordination |
| Docker | Application containerization |
| Cloud Infrastructure | Deployment and remote hosting |

## 🔍 How It Works

1. **Upload an Image:** The user submits an image of a clothing item through the application.
2. **Image Preprocessing:** The image is prepared for inference using the preprocessing steps required by the PyTorch model.
3. **Feature Extraction:** The model analyzes the image and generates visual representations of the clothing.
4. **Similarity Search:** The extracted features are compared against representations of clothing items in the dataset to identify visually similar candidates.
5. **Background Processing:** Celery workers handle supported asynchronous processing tasks, coordinated through Redis.
6. **Return Results:** The application returns matching or similar clothing items through the API for the user to explore.

## 🏗️ Architecture

The application separates API handling, machine learning inference, and background processing into distinct components.

```text
                  User
                   |
                   v
             Image Upload
                   |
                   v
              FastAPI API
                   |
                   v
           Image Preprocessing
                   |
                   v
          PyTorch ML Pipeline
                   |
                   v
          Visual Similarity Search
                   |
                   v
          Similar Clothing Results

        Asynchronous Task Processing

              FastAPI API
                   |
                   v
                Redis
             (Task Broker)
                   |
                   v
             Celery Workers
                   |
                   v
          Background Processing

          Deployment Infrastructure
                   |
                   v
              Docker Images
                   |
                   v
             Cloud Hosting
```

*Note: The diagram illustrates the application's conceptual architecture. The exact execution flow depends on how inference and background tasks are implemented.*

## 📊 Dataset

This project uses [Fashionpedia](https://fashionpedia.github.io/home/), a fashion dataset containing clothing images and detailed fashion annotations.

The dataset supports fashion-related computer vision tasks, including the analysis of clothing categories, attributes, and visual characteristics.

Fashionpedia provides a foundation for developing and evaluating image-based clothing discovery. The similarity-search pipeline uses the visual representations and dataset resources configured for the application.

## ⚙️ Installation and Setup

### Prerequisites

- Python
- Docker
- Redis
- Access to the Fashionpedia dataset and required model artifacts
- Environment variables configured for the application

### Environment Configuration

Create a `.env` file in the project root and configure the environment variables required by your setup.

```env
# Example configuration
REDIS_URL=redis://localhost:6379/0
```

Add any additional settings required for dataset paths, model artifacts, cloud services, and application configuration. Do not commit secrets or credentials to version control.

### Run with Docker

Build the Docker image:

```bash
docker build -t fashion-similarity-search .
```

Start the application using your configured Docker setup:

```bash
docker run --env-file .env -p 8000:8000 fashion-similarity-search
```

If Redis and Celery run in separate containers, configure their service connections and start them using the project's Docker Compose configuration or deployment setup.

### Run the API Locally

Install the project dependencies using the package manager and dependency files provided in the repository.

Start the FastAPI application:

```bash
uvicorn main:app --reload
```

Replace `main:app` with the actual FastAPI application import path if it differs.

Once running, access the interactive API documentation at:

`http://localhost:8000/docs`

## ☁️ Deployment

The application is containerized with Docker and deployed to a cloud environment. Containerization helps maintain a consistent runtime across development and deployment environments.

The deployment can include:

- A FastAPI service for API requests and image uploads.
- Redis for task queuing and coordination.
- Celery workers for asynchronous processing.
- Model and dataset artifacts required for inference.
- Cloud infrastructure for hosting and networking.

Production configuration should account for model loading, worker concurrency, resource requirements, environment variables, logging, and error handling.

## 🔮 Future Improvements

- Integrate product catalogs from online retailers to retrieve real-world shopping results.
- Improve similarity ranking using learned image embeddings and dedicated vector search.
- Incorporate clothing attributes such as color, category, pattern, and style into ranking.
- Optimize inference latency and background task throughput.
- Add a frontend for image uploads, result previews, and product exploration.
- Expand the dataset and evaluate retrieval quality using ranking metrics.

## 🎯 Goal

Build a practical, production-oriented fashion discovery system that connects computer vision with an intuitive shopping experience, allowing users to find similar clothing through images rather than text-based searches.
