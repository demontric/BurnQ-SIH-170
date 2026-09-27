import pandas as pd
from models.reason_codes import get_reason_message

def generate_deterministic_explanation(reason_codes) -> str:
    """
    Converts a list of reason codes into a deterministic, human-readable justification, in one to two sentences.
    """
    if not reason_codes or not isinstance(reason_codes, list) or len(reason_codes) == 0:
        return "Component passed screening."

    messages = [get_reason_message(code) for code in reason_codes]
    
    # Optional: Format into a cleaner multi-sentence or bulleted string based on preference
    return " ".join(messages)

def apply_explainability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies deterministic explainability to a DataFrame containing a 'reason_codes' column.
    """
    out = df.copy()
    
    if "reason_codes" not in out.columns:
        out["justification"] = "Component passed screening."
        return out
        
    out["justification"] = out["reason_codes"].apply(generate_deterministic_explanation)
    
    return out

def generate_single_justification(data: dict) -> str:
    """Generate an on-demand explanation for one screening record."""
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object containing screening results.")

    reason_codes = data.get("reason_codes", [])
    if isinstance(reason_codes, str):
        reason_codes = [reason_codes]
    if not isinstance(reason_codes, list):
        raise ValueError("reason_codes must be a list of strings.")

    return generate_deterministic_explanation(reason_codes)
