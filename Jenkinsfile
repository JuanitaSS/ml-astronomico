pipeline {
    agent any

    environment {
        IMAGE_NAME  = "ml-astronomico"
        IMAGE_TAG   = "${BUILD_NUMBER}"
        OUTPUTS_DIR = "outputs"
        VENV        = "venv"
        PYTHON      = "${VENV}/bin/python"
        PIP         = "${VENV}/bin/pip"
    }

    options {
        timestamps()
        timeout(time: 30, unit: "MINUTES")
        buildDiscarder(logRotator(numToKeepStr: "5"))
    }

    stages {

        stage("1 · Checkout") {
            steps {
                echo "Clonando repositorio..."
                checkout scm
                sh "ls -la"
            }
        }

        stage("2 · Instalar dependencias") {
            steps {
                sh '''
                python3 -m venv venv
                venv/bin/python -m pip install --upgrade pip
                venv/bin/pip install -r requirements.txt
                '''
            }
        }

        stage("3 · Validar dataset") {
            steps {
                sh """
                mkdir -p ${OUTPUTS_DIR}
                ${PYTHON} tests/test_dataset.py
                """
            }
            post {
                always {
                    archiveArtifacts artifacts: "${OUTPUTS_DIR}/test_results.json",
                                     allowEmptyArchive: true
                }
            }
        }

        stage("4 · Ejecutar pipeline ML") {
            steps {
                sh """
                ${PYTHON} src/main.py
                """
            }
        }

        stage("5 · Build Docker") {
            steps {
                sh """
                docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
                docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest
                """
            }
        }

        stage("6 · Ejecutar en Docker") {
            steps {
                sh """
                docker run --rm \
                    -v \$(pwd)/${OUTPUTS_DIR}:/app/outputs \
                    ${IMAGE_NAME}:latest
                """
            }
        }

        stage("7 · Archivar artefactos") {
            steps {
                sh "ls -lh ${OUTPUTS_DIR}/ || true"
            }
            post {
                always {
                    archiveArtifacts artifacts: "${OUTPUTS_DIR}/**/*",
                                     fingerprint: true,
                                     allowEmptyArchive: true
                }
            }
        }
    }

    post {
        success { echo "PIPELINE EXITOSO — Build #${BUILD_NUMBER}" }
        failure { echo "PIPELINE FALLIDO — Build #${BUILD_NUMBER}" }
    }
}