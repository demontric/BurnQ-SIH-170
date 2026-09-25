import logging
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = logging.getLogger(__name__)

# Initialize the modern Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_ID = "gemini-3.5-flash-lite"

# No tools are declared for this call, but explicitly disabling automatic
# function calling stops the SDK from evaluating/logging the AFC path.
GENERATION_CONFIG = types.GenerateContentConfig(
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
)


def generate_single_justification(data: dict) -> str:
    """On-demand NLG generation for a single component using the modern genai SDK."""
    if not data.get("is_anomaly") and not data.get("safety_slope_exceeded"):
        return "Pass: Trajectory is consistent with the lot."

    prompt = f"""
    You are a factory floor QA system. Write a single, professional sentence explaining why this component failed based on the following burn-in test data. 
    
    Data Context:
    - Component Lot: {data.get("Lot")}
    - 168h Value: {data.get("Value_168h")} µA 
    - Robust Z-Score: {data.get("robust_z_score")}
    - 24h Predicted Drift Rate: {data.get("predicted_drift_rate")} µA/h
    - Dynamic Safety Slope Limit: {data.get("safety_slope")} µA/h
    - Absolute Datasheet Limit: {data.get("datasheet_limit")} µA
    - Isolation Forest Trajectory Anomaly: {data.get("is_if_anomaly", False)}
    - Overall Anomaly Flag: {data.get("is_anomaly")}
    - Safety Slope Exceeded Flag: {data.get("safety_slope_exceeded")}

    Rules:
    1. You MUST begin the sentence with "Flagged for Rejection: "
    2. Use the specific numeric values provided to justify the decision in plain English.
    3. Do not use markdown formatting. Keep it to exactly one sentence.
    """

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=GENERATION_CONFIG,
        )
        return response.text.strip()
    except Exception:
        # Log the real failure (bad key, rate limit, network error, etc.)
        # instead of silently masking it behind the fallback sentence below.
        logger.exception(
            "Gemini justification generation failed for component %r",
            data.get("part_id") or data.get("ComponentID"),
        )
        return "Flagged for Rejection: Abnormal drift or statistical deviation detected in trajectory."


# Backward compatibility alias to prevent import errors across legacy modules
def generate_justification(row, median_168h=0.0, mad_168h=0.0):
    if hasattr(row, "to_dict"):
        d = row.to_dict()
    else:
        d = dict(row)
    return generate_single_justification(d)