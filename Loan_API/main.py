from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model and preprocessor
model_knn = joblib.load("model_knn.pkl")
model_rf = joblib.load("model_rf.pkl")
model_xgb = joblib.load("model_xgb.pkl")

models = [model_knn, model_rf, model_xgb]

# preprocessor = joblib.load("preprocessor.pkl")  # optional

# Define input schema
from pydantic import BaseModel, Field

class LoanFeatures(BaseModel):
    loan_amount: float 
    mortgage_due: float 
    property_value: float 
    years_on_job: float
    derogatory_reports: float
    delinquent_lines: float 
    credit_age_months: float 
    recent_inquiries: float
    credit_lines: float 
    debt_to_income: float 


@app.post("/predict")
def predict_loan_risk(features: LoanFeatures):
    data = np.array([[features.loan_amount, features.mortgage_due, features.property_value,
                      features.years_on_job, features.derogatory_reports, features.delinquent_lines,
                      features.credit_age_months, features.recent_inquiries, features.credit_lines,
                      features.debt_to_income]])
    
    # Optionally preprocess: data = preprocessor.transform(data)
    
    predictions = [model.predict(data)[0] for model in models]
    probs = [model.predict_proba(data)[0] for model in models]

    # Majority vote
    final_prediction = max(set(predictions), key=predictions.count)

    # Average probabilities
    avg_probs = np.mean(probs, axis=0).tolist()

    return {
        "individual_predictions": predictions,
        "final_prediction": int(final_prediction),
        "average_probability": avg_probs
    }

