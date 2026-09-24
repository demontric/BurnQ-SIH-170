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

## Running the Dashboard (Primary Demo UI)

We recommend using the polished React/Vite frontend for the primary QA inspector demo:
1. Start the API backend:
```bash
python server/api/main.py
```
2. In a separate terminal, start the React frontend:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

## Backend Evaluation (Debugging Only)

The legacy Streamlit dashboard is maintained strictly for backend model evaluation and debugging. Do not use this for the primary presentation.
```bash
cd server
streamlit run dashboard/app.py
```

## Architecture

- **Module A (`models/outlier_detection.py`)**: Uses robust z-score (Median Absolute Deviation) and Isolation Forests to flag statistical anomalies within a lot.
- **Module B (`models/drift_predictor.py`)**: Predicts the end-of-burn-in parameter value (168h) using a physical saturation curve fit on early measurements (0h, 24h). Flags parts exceeding a safe drift slope.
- **Explainability (`models/explainability.py`)**: Generates human-readable context for why a part was flagged.
