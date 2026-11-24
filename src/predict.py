"""Batch scoring helpers for trained PeoplePulse pipelines."""
from __future__ import annotations
import pandas as pd

def predict_dataframe(model, data: pd.DataFrame, feature_columns: list[str],
                      positive_class: str = "1", threshold: float = 0.5) -> pd.DataFrame:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    missing = [column for column in feature_columns if column not in data.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    if data.empty:
        raise ValueError("The prediction file contains no rows.")
    features = data[feature_columns].copy()
    probabilities = model.predict_proba(features)[:, 1]
    result = data.copy()
    result["positive_class_probability"] = probabilities.round(4)
    result["predicted_class"] = (probabilities >= threshold).astype(int)
    result["risk_band"] = pd.cut(
        probabilities, bins=[-0.000001, 0.30, 0.60, 1.000001],
        labels=["Lower model score", "Middle model score", "Higher model score"]
    ).astype(str)
    result["score_threshold"] = threshold
    result.attrs["positive_class"] = positive_class
    return result
