"""Parameter → Module B ONNX registry.

Drop the trained files into server/exports/ using the names below.
Until a file exists, get_drift_model returns None and the predictor
falls back to the 0h/24h saturation curve.
"""

import os
import onnxruntime as ort

EXPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "exports"))

DRIFT_MODELS = {
    "leakage_current": "drift_leakage.onnx",
    "iddq": "drift_iddq.onnx",
    "prop_delay": "drift_prop_delay.onnx",
}

_PARAMETER_ALIASES = {
    "leakage": "leakage_current",
    "leakagecurrent": "leakage_current",
    "leakage_current": "leakage_current",
    "iddq": "iddq",
    "prop_delay": "prop_delay",
    "propdelay": "prop_delay",
    "propagation_delay": "prop_delay",
    "propagationdelay": "prop_delay",
}

_SESSIONS = {}


def normalize_parameter(parameter) -> str:
    if parameter is None:
        raise ValueError("parameter is required for Module B")
    key = str(parameter).strip().lower().replace("-", "_").replace(" ", "_")
    key = key.replace("__", "_")
    canonical = _PARAMETER_ALIASES.get(key, key)
    if canonical not in DRIFT_MODELS:
        known = ", ".join(sorted(DRIFT_MODELS))
        raise ValueError(f"Unknown parameter '{parameter}'. Expected one of: {known}")
    return canonical


def drift_model_path(parameter) -> str:
    filename = DRIFT_MODELS[normalize_parameter(parameter)]
    return os.path.join(EXPORTS_DIR, filename)


def get_drift_model(parameter):
    """Return a cached ONNX session for this parameter, or None if not exported yet."""
    path = drift_model_path(parameter)
    if not os.path.isfile(path):
        return None
    session = _SESSIONS.get(path)
    if session is None:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        _SESSIONS[path] = session
    return session
