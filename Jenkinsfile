pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/Zahra-7696/customer-bank-churn-prediction-mlops-pipeline.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                powershell '''
                python -m pip install --upgrade pip
                pip install -r requirements.txt
                pip install pytest
                '''
            }
        }

        stage('Train Model') {
            steps {
                powershell '''
                python src/training/train_model.py
                '''
            }
        }

        stage('Run Tests') {
            steps {
                powershell '''
                pytest
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                powershell '''
                docker build -t bank-churn-api .
                '''
            }
        }

        stage('Deploy Container') {
            steps {
                powershell '''
                $container = docker ps -a --format "{{.Names}}" | Select-String "^bank-churn-container$"
                if ($container) {
                    docker rm -f bank-churn-container
                }
                docker run -d --name bank-churn-container -p 5001:5001 bank-churn-api
                '''
            }
        }
    }
}
