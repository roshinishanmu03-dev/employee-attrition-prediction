"""Lightweight dataset profiling for the PeoplePulse dashboard."""
import pandas as pd

def profile_dataset(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_percent": float(df.isna().mean().mean() * 100) if df.size else 0.0,
        "constant_columns": [str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1],
        "columns": pd.DataFrame({
            "column": df.columns.astype(str),
            "dtype": df.dtypes.astype(str).values,
            "missing_%": (df.isna().mean().values * 100).round(1),
            "unique_values": df.nunique(dropna=True).values,
        }),
    }
