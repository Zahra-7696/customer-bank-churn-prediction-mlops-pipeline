# Bank Churn Prediction - MLOps Project

This project implements an end-to-end Machine Learning Operations (MLOps) workflow for predicting customer churn in a banking dataset. It demonstrates the full lifecycle of a machine learning system, from baseline model development and experiment tracking to deployment through a Flask web application, Docker containerization, Jenkins-based CI/CD automation, and GitHub Actions-based cloud CI validation.

## Overview

Customer churn prediction is an important business problem in the banking industry because losing customers directly affects revenue and long-term growth. This project builds a machine learning system that predicts whether a customer is likely to leave the bank based on demographic and financial features.
The project was developed progressively, starting with baseline model training, followed by experiment tracking with MLflow, model registration, API development, Docker-based deployment, Jenkins pipeline automation, and GitHub Actions CI validation. The final system provides both a browser-based prediction interface and a containerized deployment workflow that can be triggered through CI/CD.


## Key Features

- Trained and compared multiple classification models, including Logistic Regression and Random Forest
- Evaluated model performance using accuracy, precision, recall, and F1-score
- Tracked experiments and metrics with MLflow
- Registered the best-performing model in the MLflow model registry
- Saved the final trained model and feature schema for deployment
- Built a Flask-based web application for customer churn prediction
- Added a cleaner homepage using Flask templates and static CSS
- Containerized the application using Docker for reproducible deployment
- Added Jenkins CI/CD automation for training, testing, image building, and container deployment
- Added GitHub Actions CI workflow for automatic validation on push and pull request
- Added a test stage using `pytest` for basic project validation

## Project Structure

```text
customer-bank-churn-prediction-mlops-pipeline/
│
├── data/                       # Raw dataset
├── models/                     # Saved trained model and feature columns
│   ├── final_model.joblib
│   └── training_columns.joblib
│
├── notebooks/                  # Development notebooks
│   ├── 01_train_baseline.ipynb
│   ├── 02_mlflow_train.ipynb
│   └── 03_register_model.ipynb
│
├── templates/
│   └── index.html              # Homepage template
│
├── static/
│   └── style.css               # Homepage styling
│
├── src/
│   ├── api/
│   │   └── app.py              # Flask application
│   └── training/
│       └── train_model.py      # Separate training script
│
├── tests/
│   └── test_files.py           # Basic validation tests
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile                  # Docker image definition
├── Jenkinsfile                 # Jenkins CI/CD pipeline
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

## Technologies Used

Python, Pandas, Scikit-learn, MLflow, Flask, Docker, Jenkins, GitHub Actions, Pytest, Joblib

## Machine Learning Workflow

The machine learning pipeline begins with loading and preprocessing the banking churn dataset. Categorical variables are transformed using one-hot encoding so that they can be used by machine learning algorithms. Multiple models were trained and evaluated, including Logistic Regression as a baseline model and Random Forest as a stronger ensemble model.

After comparing performance, the Random Forest model was selected as the final deployment model because it achieved better results across the main classification metrics. Experiments were tracked in MLflow, and the selected model was registered before being exported as a deployment-ready artifact.

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
```text
POST /predict-form
```

For browser-based interaction, the homepage form sends user input to the Flask application, which preprocesses the values, aligns them with the saved training feature columns, and returns the prediction result.

## Running the Project Locally

python src/training/train_model.py
python src/api/app.py
http://127.0.0.1:5001

## Running with Docker

docker build -t bank-churn-api .
docker run -p 5001:5001 bank-churn-api

## Jenkins CI/CD Pipeline

This project includes a Jenkins pipeline that automates the core ML deployment workflow. The Jenkins pipeline reduces manual deployment steps and helps ensure that code updates remain testable and deployable. It also demonstrates how a machine learning application can be integrated into a more realistic CI/CD workflow rather than being run only by hand.

### Jenkins pipeline stages
- Checkout source code from GitHub
- Set up a project-specific Python virtual environment
- Install dependencies from `requirements.txt`
- Train the model using `src/training/train_model.py`
- Run tests with `pytest`
- Build the Docker image
- Deploy the updated container

### GitHub Actions CI Workflow
Stages:
- Checkout repository
- Setup Python
- Create virtual environment
- Install dependencies
- Train model
- Run tests
- Build Docker image
Purpose:
GitHub Actions validates the project automatically on every push and pull request in a clean cloud environment.



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
- Jenkins-based CI/CD automation
- GitHub Actions CI Workflow
- Basic automated testing

These topics are especially relevant for machine learning engineering, data science, MLOps, and applied AI roles.


## Future Improvements

Possible future enhancements include:

- Add API tests
- Add CI badges
- Add cloud deployment


## Author

**Zahra S. Torabi**  
PhD in Computer Science with experience in machine learning, optimization, and applied AI systems.  
Contact: z.torabi.university@gmail.com
