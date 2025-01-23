import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI()

model = None
data = None
scaler = MinMaxScaler()

class PredictionInput(BaseModel):
    ProductionVolume: float
    ProductionCost: float
    SupplierQuality: float
    DeliveryDelay: float
    DefectRate: float
    QualityScore: float
    MaintenanceHours: float
    DowntimePercentage: float
    InventoryTurnover: float
    StockoutRate: float
    WorkerProductivity: float
    SafetyIncidents: float
    EnergyConsumption: float
    EnergyEfficiency: float
    AdditiveProcessTime: float
    AdditiveMaterialCost: float

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global data

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        data = pd.read_csv(file.file)
        return {"message": "File uploaded successfully.", "columns": list(data.columns)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

@app.post("/train")
def train_model():
    global model, data, scaler

    if data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a dataset first.")

    try:
        x = data.iloc[:, :-1].values  # Features
        y = data.iloc[:, -1].values   # Target

        # Scale the data
        x = scaler.fit_transform(x)

        # Split the data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

        # Handle class imbalance using SMOTE
        smote = SMOTE()
        x_train, y_train = smote.fit_resample(x_train, y_train)

        # Train the Random Forest model
        model = RandomForestClassifier(random_state=42)
        model.fit(x_train, y_train)

        pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average='weighted')
        recall = recall_score(y_test, pred, average='weighted')
        f1 = f1_score(y_test, pred, average='weighted')

        return {
            "message": "Model trained successfully.",
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.post("/predict")
def predict(input_data: PredictionInput):
    global model, scaler

    if model is None:
        raise HTTPException(status_code=400, detail="Model is not trained. Please train the model first.")

    try:
        input_df = pd.DataFrame([input_data.dict().values()], columns=input_data.dict().keys())
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        confidence = model.predict_proba(input_scaled).max()

        return {"DefectStatus": int(prediction[0]), "Confidence": round(confidence, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
