pipeline {
    agent any

    stages {
        stage('Set Up Virtual Environment') {
            steps {
                powershell '''
                python -m venv .venv
                .\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
                .\\.venv\\Scripts\\pip.exe install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                powershell '''
                .\\.venv\\Scripts\\python.exe src/training/train_model.py
                '''
            }
        }

        stage('Run Tests') {
            steps {
                powershell '''
                $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
                .\\.venv\\Scripts\\python.exe -m pytest tests/test_files.py
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