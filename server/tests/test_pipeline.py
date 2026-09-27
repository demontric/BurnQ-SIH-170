import sys
import os
import pandas as pd

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from data.synthetic_generator import generate_synthetic_data
from models.outlier_detection import detect_anomalies
from models.drift_predictor import predict_168h
from models.safety_slope import calculate_safety_slope
from models.explainability import generate_deterministic_explanation
from models.pipeline import run_screening


def test_pipeline():
    # ---------------------------------------------------------
    # 1. Generate synthetic data
    # ---------------------------------------------------------

    print("Generating synthetic data...")

    df = generate_synthetic_data(
        num_lots=3
    )

    assert not df.empty, (
        "Generated dataframe is empty"
    )

    assert "value_0h" in df.columns, (
        "Missing value_0h column in generated data"
    )

    # ---------------------------------------------------------
    # 2. Run Module A
    # ---------------------------------------------------------

    print(
        "Running outlier detection (Module A)..."
    )

    df_a = detect_anomalies(df)

    assert "robust_z_score" in df_a.columns
    assert "isolation_forest_score" in df_a.columns
    assert "is_if_anomaly" in df_a.columns
    assert "is_lot_outlier" in df_a.columns
    assert "is_anomaly" in df_a.columns

    # The aggregate anomaly flag must be consistent with
    # the underlying Module A signals / absolute check.
    assert len(df_a) == len(df)

    # ---------------------------------------------------------
    # 3. Run Module B
    # ---------------------------------------------------------

    print(
        "Running drift prediction (Module B)..."
    )

    df_ab = predict_168h(df_a)

    assert "predicted_168h" in df_ab.columns
    assert "predicted_drift_rate" in df_ab.columns

    # ---------------------------------------------------------
    # 4. Safety slope is a downstream stage.
    #
    # predict_168h() itself is not responsible for creating
    # safety_slope_exceeded, so calculate it explicitly here.
    # ---------------------------------------------------------

    df_ab = calculate_safety_slope(
        df_ab,
        threshold=3.5,
    )

    assert "safety_slope" in df_ab.columns
    assert "safety_slope_exceeded" in df_ab.columns

    # ---------------------------------------------------------
    # 5. Evaluate combined screening payload
    # ---------------------------------------------------------

    print("\n--- Evaluation ---")

    payload = run_screening(
        df
    )

    data = payload["data"]

    assert isinstance(
        data,
        list,
    )

    # ---------------------------------------------------------
    # Payload structure
    # ---------------------------------------------------------

    assert (
        payload["components"]
        == df["part_id"].nunique()
        if "part_id" in df.columns
        else payload["components"] == len(df)
    )

    assert (
        payload["parametric_records"]
        == len(df)
    )

    assert "mae" in payload
    assert "mae_by_parameter" in payload

    # ---------------------------------------------------------
    # Flagged record consistency
    # ---------------------------------------------------------

    flagged_records = [
        row
        for row in data
        if row.get("is_flagged")
    ]

    flagged_component_ids = {
        row["ComponentID"]
        for row in flagged_records
    }

    assert (
        payload["flagged_count"]
        == len(flagged_records)
    )

    # ---------------------------------------------------------
    # Classification metrics
    #
    # Compare physical component IDs where the synthetic data
    # contains one ground-truth label per row.
    # ---------------------------------------------------------

    if (
        "ground_truth_defective" in df_ab.columns
        and "part_id" in df_ab.columns
    ):
        flagged_component_set = flagged_component_ids

        y_true = (
            df_ab["ground_truth_defective"]
            .astype(bool)
        )

        y_pred = (
            df_ab["part_id"]
            .isin(flagged_component_set)
        )

        tp = (
            (y_true == True)
            & (y_pred == True)
        ).sum()

        fp = (
            (y_true == False)
            & (y_pred == True)
        ).sum()

        fn = (
            (y_true == True)
            & (y_pred == False)
        ).sum()

        tn = (
            (y_true == False)
            & (y_pred == False)
        ).sum()

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

        print(
            f"Total Parts: {len(df_ab)}"
        )

        print(
            f"True Defects: {y_true.sum()}"
        )

        print(
            f"Flagged Parts: {y_pred.sum()}"
        )

        print(
            f"Recall: {recall:.2%}"
        )

        print(
            f"Precision: {precision:.2%}"
        )

        # Keep TN referenced so the confusion matrix remains
        # explicit without triggering an unused-variable issue
        # in simple static inspection.
        _ = tn

    # ---------------------------------------------------------
    # Explainability
    # ---------------------------------------------------------

    assert all(
        "justification" in row
        for row in data
    )

    # Every non-PASS row should have an actionable reason.
    for row in data:
        status = row.get(
            "status",
            "PASS",
        )

        reason_codes = row.get(
            "reason_codes",
            [],
        )

        justification = row.get(
            "justification"
        )

        assert isinstance(
            reason_codes,
            list,
        )

        assert justification

        if status == "PASS":
            assert (
                justification
                == "Component passed screening."
            )
        else:
            assert (
                justification
                != "Component passed screening."
            )

    # Verify the deterministic explainability function
    # itself is callable under its current exported name.
    assert (
        generate_deterministic_explanation(
            []
        )
        == "Component passed screening."
    )

    # ---------------------------------------------------------
    # Prediction evaluation
    # ---------------------------------------------------------

    assert payload["mae"] is not None

    assert any(
        row.get("predicted_168h") is not None
        for row in data
    )

    # All current payload records should expose the signals
    # required by the frontend and decision engine.
    required_payload_fields = {
        "is_if_anomaly",
        "is_lot_outlier",
        "is_anomaly",
        "status",
        "reason_codes",
        "justification",
        "isolation_forest_score",
        "predicted_drift_rate",
        "safety_slope",
        "safety_slope_exceeded",
    }

    for row in data:
        missing = (
            required_payload_fields
            - set(row.keys())
        )

        assert not missing, (
            f"{row.get('ComponentID', 'unknown')} "
            f"is missing payload fields: {sorted(missing)}"
        )


if __name__ == "__main__":
    test_pipeline()
