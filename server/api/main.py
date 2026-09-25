import os
import sys
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
from models.explainability import generate_single_justification

# Ensure the parent directory is in the path for model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pipeline import run_screening
from models.drift_predictor import predict_168h

# Initialize FastAPI App
app = FastAPI(
    title="Burn-In Anomaly Detector API",
    description="API for AI-Driven Anomaly Detection in Component Burn-In & Screening",
    version="1.0.0"
)

# Configure CORS for React/Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health Check"])
def read_root():
    """Health check endpoint to verify the API is running."""
    return {"status": "ok", "message": "Burn-In Anomaly Detector API is running."}

@app.post("/api/detect-anomaly", tags=["Pipeline"])
@app.post("/detect-anomaly", tags=["Pipeline"])
async def detect_anomaly_endpoint(
    file: UploadFile = File(...),
    risk_tolerance: float = Form(50.0),
    datasheet_limit: float = Form(50.0)
):
    """
    Main Screening Pipeline (Gate 1 + Gate 2 + Explainability).
    Accepts a CSV file, risk tolerance, and datasheet limit.
    Returns a JSON payload with predictions, anomaly flags, and NLG justifications.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Execute the end-to-end multi-gate screening pipeline
        result = run_screening(
            df=df, 
            datasheet_limit=datasheet_limit, 
            risk_tolerance=risk_tolerance
        )
        
        return JSONResponse(content=result)
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/predict-drift", tags=["Standalone Modules"])
async def predict_drift_endpoint(file: UploadFile = File(...)):
    """
    Standalone Module B execution (Forecast 168h).
    Useful for isolated testing of the regression model without the full pipeline.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        req_cols = ['value_0h', 'value_24h', 'value_96h', 'parameter']
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
             raise ValueError(f"Missing required columns: {missing}")
        
        # Run standalone drift prediction
        result_df = predict_168h(df)
        
        return JSONResponse(content=result_df.to_dict(orient="records"))
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/api/explain", tags=["Explainability"])
async def explain_anomaly(request: Request):
    """Generates an on-demand LLM explanation for a single row."""
    try:
        data = await request.json()
        explanation = generate_single_justification(data)
        return {"justification": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)