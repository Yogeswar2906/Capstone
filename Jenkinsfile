pipeline {
    agent any

    environment {
        // Set a project name for Docker Compose to keep container names consistent
        COMPOSE_PROJECT_NAME = 'smart_attendance_system'
    }

    stages {
        stage('Checkout') {
            steps {
                // Check out the code from the Git repository config
                checkout scm
            }
        }

        stage('Build') {
            steps {
                echo 'Building Docker images...'
                script {
                    if (isUnix()) {
                        sh 'docker compose build'
                    } else {
                        bat 'docker compose build'
                    }
                }
            }
        }

        stage('Deploy') {
            steps {
                echo 'Deploying application containers...'
                script {
                    if (isUnix()) {
                        sh 'docker compose down'
                        sh 'docker compose up -d'
                    } else {
                        bat 'docker compose down'
                        bat 'docker compose up -d'
                    }
                }
                echo 'Application started successfully!'
            }
        }

        stage('Clean') {
            steps {
                echo 'Cleaning up dangling Docker images...'
                script {
                    if (isUnix()) {
                        sh 'docker image prune -f'
                    } else {
                        bat 'docker image prune -f'
                    }
                }
            }
        }
    }

    post {
        success {
            echo 'Pipeline completed successfully! You can access the app at http://localhost:8080'
        }
        failure {
            echo 'Pipeline failed. Please check stage logs for details.'
        }
    }
}
