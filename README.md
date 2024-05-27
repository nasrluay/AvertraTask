Introduction

This project leverages Large Language Models (LLMs) to enhance utility operations by integrating a pre-trained model into a chatbot. The chatbot is designed to improve internal development, business processes, and customer experience by utilizing accumulated knowledge and providing accurate, contextually relevant responses.
Project Structure

The project consists of two main components:

    Admin Interface: For managing document processing and vector store creation.
    User Interface: For querying the chatbot and receiving responses.

Prerequisites

    Python 3.8+
    Docker
    AWS CLI configured with appropriate credentials


AWS Configuration
Ensure your AWS CLI is configured correctly:

    Install the AWS CLI: AWS CLI Installation Guide
    Configure your AWS CLI with your credentials:

    aws configure

        AWS Access Key ID: YOUR_ACCESS_KEY
        AWS Secret Access Key: YOUR_SECRET_KEY
        Default region name: us-east-1 (or your preferred region)
        Default output format: json


Running the Docker Containers
Admin Interface

    Build the Docker image:
        docker build -t utility-reader-admin .

    Run the Docker container:
        docker run -e BUCKET_NAME=avertra-utility-bot -e AWS_REGION=us-east-1 -v C:/Users/gluay/.aws:/root/.aws -p 8083:8083 -it utility-reader-admin

User Interface

    Build the Docker image:
        docker build -t utility-reader-client .
    Run the Docker container:
        docker run -e BUCKET_NAME=avertra-utility-bot -e AWS_REGION=us-east-1 -v C:/Users/gluay/.aws:/root/.aws -p 8084:8084 -it utility-reader-client
