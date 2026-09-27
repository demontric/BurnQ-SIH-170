import io
import os
import sys
import traceback

import pandas as pd

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Request,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# Ensure server/ is on Python's import path.
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


from models.explainability import (
    generate_single_justification,
)

from models.pipeline import (
    run_screening,
)

from models.drift_predictor import (
    predict_168h,
)


# -------------------------------------------------------------
# FastAPI application
# -------------------------------------------------------------

app = FastAPI(
    title="Burn-In Anomaly Detector API",
    description=(
        "API for AI-Driven Anomaly Detection "
        "in Component Burn-In & Screening"
    ),
    version="1.0.0",
)


# -------------------------------------------------------------
# CORS
# -------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],

    allow_credentials=True,

    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],

    allow_headers=[
        "*"
    ],
)


# -------------------------------------------------------------
# Health check
# -------------------------------------------------------------

@app.get(
    "/",
    tags=["Health Check"],
)
def read_root():
    return {
        "status": "ok",
        "message": (
            "Burn-In Anomaly Detector API "
            "is running."
        ),
    }


# -------------------------------------------------------------
# Main detection endpoint
# -------------------------------------------------------------

@app.post(
    "/api/detect-anomaly",
    tags=["Pipeline"],
)
@app.post(
    "/detect-anomaly",
    tags=["Pipeline"],
)
async def detect_anomaly_endpoint(
    file: UploadFile = File(...),
    risk_tolerance: float = Form(50.0),
    datasheet_limit: float = Form(50.0),
):
    """
    Main end-to-end screening pipeline.

    Accepts:
        - CSV file
        - risk tolerance
        - datasheet limit

    Returns:
        - Module B prediction
        - Module A anomaly detection
        - safety slope
        - decision
        - deterministic explanation
        - evaluation metrics
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A CSV file is required.",
        )

    if not file.filename.lower().endswith(
        ".csv"
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Only CSV files are supported."
            ),
        )

    try:
        contents = await file.read()

        if not contents:
            raise ValueError(
                "Uploaded CSV file is empty."
            )

        df = pd.read_csv(
            io.BytesIO(contents)
        )

        if df.empty:
            raise ValueError(
                "Uploaded CSV contains no rows."
            )

        result = run_screening(
            df=df,
            datasheet_limit=datasheet_limit,
            risk_tolerance=risk_tolerance,
        )

        return JSONResponse(
            content=result
        )

    except ValueError as exc:
        # Print full traceback during development.
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        # This is important while debugging:
        # the terminal will now show the exact file/line.
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Internal Server Error: "
                f"{str(exc)}"
            ),
        )


# -------------------------------------------------------------
# Standalone Module B endpoint
# -------------------------------------------------------------

@app.post(
    "/predict-drift",
    tags=["Standalone Modules"],
)
async def predict_drift_endpoint(
    file: UploadFile = File(...),
):
    """
    Standalone Module B execution.

    Forecasts 168h from 0h/24h information only.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A CSV file is required.",
        )

    if not file.filename.lower().endswith(
        ".csv"
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Only CSV files are supported."
            ),
        )

    try:
        contents = await file.read()

        if not contents:
            raise ValueError(
                "Uploaded CSV file is empty."
            )

        df = pd.read_csv(
            io.BytesIO(contents)
        )

        required_columns = [
            "value_0h",
            "value_24h",
            "parameter",
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        result_df = predict_168h(
            df
        )

        # Convert NumPy scalars/NaNs safely.
        result_df = result_df.where(
            pd.notna(result_df),
            None,
        )

        records = (
            result_df
            .to_dict(
                orient="records"
            )
        )

        return JSONResponse(
            content=records
        )

    except ValueError as exc:
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Internal Server Error: "
                f"{str(exc)}"
            ),
        )


# -------------------------------------------------------------
# Explainability endpoint
# -------------------------------------------------------------

@app.post(
    "/api/explain",
    tags=["Explainability"],
)
async def explain_anomaly(
    request: Request,
):
    """
    Generate a deterministic explanation
    for a single screening result.
    """

    try:
        data = await request.json()

        explanation = (
            generate_single_justification(
                data
            )
        )

        return {
            "justification": explanation
        }

    except ValueError as exc:
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# -------------------------------------------------------------
# Local execution
# -------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )