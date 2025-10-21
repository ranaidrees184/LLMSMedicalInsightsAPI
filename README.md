Health Insight API

This repository contains the FastAPI-based backend for the Health Insight System.

Getting Started
Follow these steps to get a local copy up and running. 📋 Prerequisites

Make sure you have the following installed:

Python 3.9+
fastapi
pip (Python package manager)
(Optional) Git for cloning the repository
Installation

Clone the Repository git clone https://github.com/ranaidrees184/LLMSMedicalInsightAPI.git cd LLMSMedicalInsightAPI

Create a Virtual Environment (Recommended) python -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate

Install Dependencies pip install -r requirements.txt

Run the Application uvicorn app.main:app --reload

The API will start locally at: 👉 http://127.0.0.1:8000

Interactive API docs are available at: 👉 http://127.0.0.1:8000/docs

Deployment (General Instructions)

This API can run on any system that supports Python. To deploy it:

Copy all project files to your server or environment.

Install dependencies with pip install -r requirements.txt.

Start the FastAPI app:

uvicorn app:app --host 0.0.0.0 --port 8000

Ensure port 8000 (or your chosen port) is open for access.

No additional setup, keys, or dependencies are required.

Testing the API

Once running, test endpoints via:

Browser Swagger UI: http://127.0.0.1:8000/docs

Curl/Postman: Send requests manually for validation.
