# AI-Driven Anomaly Detection in Component Burn-In & Screening

This is a prototype web app for detecting latent defects during burn-in stress testing of electronic components. It identifies statistical outliers and abnormal drift trajectories even when components remain within absolute datasheet specifications.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Data:**
   (Optional: The dashboard will auto-generate this if missing)
   ```bash
   python data/synthetic_generator.py
   ```

3. **Run tests:**
   ```bash
   python tests/test_pipeline.py
   ```

## Running the Dashboard

Launch the Streamlit single-page app:
```bash
streamlit run dashboard/app.py
```

## Running the API (Optional)

The backend features FastAPI endpoints for remote inference:
```bash
python api/main.py
```
Test with: `curl -X POST -F "file=@data/synthetic_burnin_data.csv" http://localhost:8000/detect-anomaly`

## Architecture

- **Module A (`models/outlier_detection.py`)**: Uses robust z-score (Median Absolute Deviation) and Isolation Forests to flag statistical anomalies within a lot.
- **Module B (`models/drift_predictor.py`)**: Predicts the end-of-burn-in parameter value (168h) using a physical saturation curve fit on early measurements (0h, 24h). Flags parts exceeding a safe drift slope.
- **Explainability (`models/explainability.py`)**: Generates human-readable context for why a part was flagged.
