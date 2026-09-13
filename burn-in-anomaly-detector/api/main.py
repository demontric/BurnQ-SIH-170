from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
import sys
import os

# Add parent dir to path so we can import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.outlier_detection import detect_anomalies
from models.drift_predictor import predict_168h

app = FastAPI(title="Burn-In Anomaly Detector API")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Burn-In Anomaly Detector API is running."}

@app.post("/detect-anomaly")
async def detect_anomaly_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validate columns
        req_cols = ['lot_id', 'parameter', 'value_0h', 'value_24h', 'value_96h', 'value_168h']
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
             raise ValueError(f"Missing required columns: {missing}")
        
        # Run module A
        result_df = detect_anomalies(df)
        
        # Return as JSON records
        return JSONResponse(content=result_df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-drift")
async def predict_drift_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        req_cols = ['value_0h', 'value_24h', 'parameter']
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
             raise ValueError(f"Missing required columns: {missing}")
        
        # Run module B
        result_df = predict_168h(df)
        
        return JSONResponse(content=result_df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
