# Predictive Pulse — Hypertension Intelligence System

Advanced machine learning web application for hypertension stage prediction using 13 clinical parameters.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (optional — logreg_model.pkl is included)
python train_model.py

# Run application
python app.py
```

Then open: http://localhost:5000

## Features
- 7 ML algorithms evaluated (Logistic Regression selected — 100% accuracy)
- 4-stage hypertension classification: Normal, Stage-1, Stage-2, Hypertensive Crisis
- Personalised clinical recommendations per prediction
- Confidence scoring with probability estimates
- Dark clinical medical-grade UI with real-time validation

## Dataset
- 1,825 patient records from Kaggle
- 13 features: demographics, medical history, symptoms, vitals, lifestyle

## Tech Stack
- Backend: Python, Flask, Scikit-learn, Joblib
- Frontend: HTML5, CSS3, Vanilla JS (zero dependencies)
- Model: Logistic Regression (logreg_model.pkl)

## Disclaimer
For educational and decision-support purposes only. Not a substitute for professional medical diagnosis.
