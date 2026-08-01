# Heart Disease Prediction - ML Model Deployment

## Project Overview
This project predicts whether a patient is at risk of heart disease based on clinical parameters. A **Random Forest Classifier** is used for prediction. The trained model is served through a **Flask REST API** and deployed on **Render** as a live web service.

## Dataset
- **Source:** [Heart Disease Dataset - Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Records:** 1025
- **Features:** 13 clinical parameters
- **Target:** Heart disease presence (1 = Yes, 0 = No)

## Features Used
| Feature   | Description                                      |
|-----------|--------------------------------------------------|
| age       | Age of the patient                               |
| sex       | Gender (1 = Male, 0 = Female)                    |
| cp        | Chest pain type (0-3)                            |
| trestbps  | Resting blood pressure (mm Hg)                   |
| chol      | Serum cholesterol (mg/dl)                        |
| fbs       | Fasting blood sugar > 120 mg/dl (1=True, 0=False)|
| restecg   | Resting ECG results (0-2)                        |
| thalach   | Maximum heart rate achieved                      |
| exang     | Exercise induced angina (1=Yes, 0=No)            |
| oldpeak   | ST depression induced by exercise                |
| slope     | Slope of peak exercise ST segment (0-2)          |
| ca        | Number of major vessels colored by flourosopy (0-4)|
| thal      | Thalassemia (0-3)                                |

## Model Details
- **Algorithm:** Random Forest Classifier
- **Training/Testing Split:** 80% / 20%
- **Accuracy:** ~80.49%

## Project Structure
```
HeartDiseaseDeployment/
│
├── app.py               # Flask REST API
├── model.pkl            # Trained ML model
├── train_model.py       # Model training script
├── heart.csv            # Dataset
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── .dockerignore        # Docker ignore file
├── README.md            # Project documentation
└── templates/
    └── index.html       # Web interface
```

## How to Run Locally

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the model
```bash
python train_model.py
```

### Step 3: Run the Flask app
```bash
python app.py
```

### Step 4: Open in browser
Go to `http://localhost:5000`

## API Usage

### Endpoint
`POST /predict`

### Request (JSON)
```json
{
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 0,
    "thal": 2
}
```

### Response
```json
{
    "prediction": "Heart Disease Detected"
}
```

## Deployed Application
- **Render URL:** _[Add your Render deployment URL here]_

## Conclusion
The Random Forest Classifier achieved an accuracy of approximately 80.49% on the heart disease prediction task. The model was successfully deployed as a REST API using Flask and containerized with Docker for cloud deployment on Render. During the development process, challenges were faced in setting up the correct dependencies and ensuring the model file was properly loaded in the production environment. This project highlights the importance of MLOps practices in machine learning, as they bridge the gap between model development and deployment, ensuring that ML models are not only accurate but also accessible and maintainable in real-world applications.

## Author
AI-ML Assignment 10 - End-to-End ML Model Deployment
