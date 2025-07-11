from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model and preprocessor
model = joblib.load("loan_model.pkl")
# preprocessor = joblib.load("preprocessor.pkl")  # optional

# Define input schema
from pydantic import BaseModel, Field

class LoanFeatures(BaseModel):
    loan_amount: float = Field(..., alias="LOAN")
    mortgage_due: float = Field(..., alias="MORTDUE")
    property_value: float = Field(..., alias="VALUE")
    years_on_job: float = Field(..., alias="YOJ")
    derogatory_reports: float = Field(..., alias="DEROG")
    delinquent_lines: float = Field(..., alias="DELINQ")
    credit_age_months: float = Field(..., alias="CLAGE")
    recent_inquiries: float = Field(..., alias="NINQ")
    credit_lines: float = Field(..., alias="CLNO")
    debt_to_income: float = Field(..., alias="DEBTINC")


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
