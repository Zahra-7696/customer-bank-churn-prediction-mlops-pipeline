# Bank Churn Prediction - MLOps Project

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for predicting customer churn in a banking dataset. It demonstrates the full lifecycle of a machine learning system, from model development and experiment tracking to deployment through a Flask web application and Docker containerization.

## Overview

Customer churn prediction is an important business problem in the banking industry because losing customers directly affects revenue and long-term growth. This project builds a machine learning system that predicts whether a customer is likely to leave the bank based on demographic and financial features.

The project was developed progressively, starting with baseline model training, followed by experiment tracking with MLflow, model registration, API creation, and finally Docker-based deployment. The final system provides both a prediction endpoint and a simple web interface for user interaction.

## Key Features

- Trained and compared multiple classification models, including Logistic Regression and Random Forest
- Evaluated model performance using accuracy, precision, recall, and F1-score
- Tracked experiments and metrics with MLflow
- Registered the best-performing model in the MLflow model registry
- Saved the final trained model and feature schema for deployment
- Built a Flask-based web application for customer churn prediction
- Added a user-friendly homepage form for entering customer data
- Containerized the application using Docker for reproducible deployment

## Project Structure

```text
bank-churn-mlops/
│
├── data/                       # Raw dataset
├── models/                     # Saved trained model and feature columns
│   ├── final_model.joblib
│   └── training_columns.joblib
│
├── notebooks/                  # Development and learning notebooks
│   ├── 01_train_baseline.ipynb
│   ├── 02_mlflow_train.ipynb
│   ├── 03_register_model.ipynb
│   └── 04_model_api.ipynb
│
├── src/
│   └── api/
│       └── app.py              # Flask application
│
├── Dockerfile                  # Docker image definition
├── .dockerignore
├── requirements.txt
├── mlflow.db
└── README.md
```

## Technologies Used

- Python
- Pandas
- Scikit-learn
- MLflow
- Flask
- Docker
- Joblib

## Machine Learning Workflow

The machine learning pipeline begins with loading and preprocessing the banking churn dataset. Categorical variables are transformed using one-hot encoding so that they can be used by machine learning algorithms. Multiple models were trained and evaluated, including Logistic Regression as a baseline model and Random Forest as a stronger ensemble model.

After comparing performance, the Random Forest model was selected as the final model because it achieved better results across the main classification metrics. Experiments were tracked in MLflow, and the selected model was registered before being exported as a deployment-ready artifact.

## Model Performance Summary

The Random Forest model outperformed Logistic Regression and was chosen as the final deployment model.

| Model                | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|----------|--------|----------|
| Logistic Regression | ~0.80    | Lower    | Lower  | Lower    |
| Random Forest       | ~0.86    | Higher   | Better | Better   |

## Application Features

The deployed application includes two ways to interact with the model:

### 1. Web Interface
Users can open the homepage in a browser, enter customer information such as credit score, geography, age, balance, and salary, and receive a churn prediction directly on the page.

### 2. API Endpoint
The model can also be accessed programmatically through an API endpoint.

**Endpoint**
```
POST /predict-form
```

For browser-based interaction, the homepage form sends user input to the Flask application, which preprocesses the values, aligns them with the saved training feature columns, and returns the prediction result.

## Running the Project Locally

To run the application locally:

```bash
python src/api/app.py
```

Then open the following URL in your browser:

```
http://127.0.0.1:5001
```

## Running with Docker

Build the Docker image:

```bash
docker build -t bank-churn-api .
```

Run the Docker container:

```bash
docker run -p 5001:5001 bank-churn-api
```

Then open:

```
http://127.0.0.1:5001
```

## Why This Project Matters

This project is not only a machine learning model but also a demonstration of practical MLOps concepts. It shows how a model can move from experimentation to a usable deployed system. It covers several important technical areas, including:

- Machine learning classification
- Data preprocessing and feature encoding
- Model evaluation and selection
- Experiment tracking
- Model registry concepts
- Web API development
- Model serving
- Docker-based deployment

These topics are especially relevant for machine learning engineering, data science, and applied AI roles.

## Future Improvements

Possible future enhancements include:

- Adding churn probability output instead of only class labels
- Improving the front-end design using templates and static files
- Adding input validation and error handling
- Refactoring training into a separate script
- Building an end-to-end scikit-learn pipeline
- Adding CI/CD for automated testing and deployment
- Deploying the containerized application to a cloud service such as Azure, AWS, or Render

## Author

**Zahra S. Torabi**
PhD in Computer Science with experience in machine learning, optimization, and applied AI systems.
Contact: z.torabi.university@gmail.com 
