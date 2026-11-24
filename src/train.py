"""Training, evaluation and persistence utilities for PeoplePulse."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
MODEL_VERSION = "1.0.0"

def _binary_target(series: pd.Series):
    clean = series.astype("string").str.strip().str.lower()
    clean = clean[clean.notna() & clean.ne("")]
    classes = sorted(clean.unique().tolist())
    if len(classes) != 2:
        raise ValueError(f"Target must have exactly 2 classes; found {len(classes)}: {classes}")
    mapping = {classes[0]: 0, classes[1]: 1}
    return clean.map(mapping).astype(int), mapping

def _build_preprocessor(X: pd.DataFrame):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    if not transformers:
        raise ValueError("No usable feature columns were found.")
    return ColumnTransformer(transformers=transformers, remainder="drop")

def _importance(pipeline: Pipeline, source_columns: list[str]) -> pd.DataFrame:
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        weights = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        weights = np.abs(np.asarray(classifier.coef_[0], dtype=float))
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    try:
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        grouped = {}
        for name, weight in zip(names, weights):
            clean = str(name).split("__", 1)[-1]
            source = next((c for c in source_columns if clean == c or clean.startswith(c + "_")), clean)
            grouped[source] = grouped.get(source, 0.0) + float(weight)
        total = sum(grouped.values()) or 1.0
        return pd.DataFrame(sorted(((k, v / total) for k, v in grouped.items()),
            key=lambda item: item[1], reverse=True), columns=["feature", "importance"]).head(20)
    except (ValueError, AttributeError):
        return pd.DataFrame(columns=["feature", "importance"])

def train_model(data: pd.DataFrame, target_column: str, test_size: float = 0.2) -> dict[str, Any]:
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4")
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' is missing.")
    frame = data.copy()
    frame = frame.loc[frame[target_column].notna()].copy()
    frame = frame.loc[frame[target_column].astype(str).str.strip().ne("")]
    if len(frame) < 10:
        raise ValueError("At least 10 labelled rows are required.")
    y, mapping = _binary_target(frame[target_column])
    frame = frame.loc[y.index]
    X = frame.drop(columns=[target_column]).dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=False) > 1]
    if X.empty:
        raise ValueError("No usable feature columns remain after removing empty/constant columns.")
    counts = y.value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Each target class needs at least 2 rows.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }
    evaluated = {}
    for name, classifier in candidates.items():
        pipeline = Pipeline([("preprocess", _build_preprocessor(X_train)), ("classifier", classifier)])
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        probability = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
            "precision": float(precision_score(y_test, predicted, zero_division=0)),
            "recall": float(recall_score(y_test, predicted, zero_division=0)),
            "f1_score": float(f1_score(y_test, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probability)),
            "pr_auc": float(average_precision_score(y_test, probability)),
        }
        evaluated[name] = {"model": pipeline, "metrics": metrics,
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist()}
    best_name = max(evaluated, key=lambda n: evaluated[n]["metrics"]["f1_score"])
    best = evaluated[best_name]
    return {
        "model": best["model"], "model_name": best_name, "metrics": best["metrics"],
        "all_metrics": {n: v["metrics"] for n, v in evaluated.items()},
        "confusion_matrix": best["confusion_matrix"],
        "importance": _importance(best["model"], X.columns.tolist()),
        "feature_columns": X.columns.tolist(), "target_mapping": mapping,
        "positive_class": next(k for k, v in mapping.items() if v == 1),
        "test_rows": len(X_test), "train_rows": len(X_train),
        "target_column": target_column, "model_version": MODEL_VERSION,
    }

def save_model(result: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = ("model", "model_name", "feature_columns", "target_mapping",
        "positive_class", "target_column", "model_version", "metrics")
    joblib.dump({key: result[key] for key in keys}, destination)
    return destination

def load_model(path: str | Path) -> dict[str, Any]:
    return joblib.load(Path(path))
