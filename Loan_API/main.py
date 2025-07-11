from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model and preprocessor
model = joblib.load("loan_model.pkl")
# preprocessor = joblib.load("preprocessor.pkl")  # optional

# Define input schema
class LoanFeatures(BaseModel):
    LOAN: float
    MORTDUE: float
    VALUE: float
    YOJ: float
    DEROG: float
    DELINQ: float
    CLAGE: float
    NINQ: float
    CLNO: float
    DEBTINC: float

@app.post("/predict")
def predict_loan_risk(features: LoanFeatures):
    data = np.array([[features.LOAN, features.MORTDUE, features.VALUE,
                      features.YOJ, features.DEROG, features.DELINQ,
                      features.CLAGE, features.NINQ, features.CLNO,
                      features.DEBTINC]])
    
    # If using preprocessor:
    # data = preprocessor.transform(data)

    prediction = model.predict(data)
    proba = model.predict_proba(data).tolist()

    return {"prediction": int(prediction[0]), "probability": proba}
