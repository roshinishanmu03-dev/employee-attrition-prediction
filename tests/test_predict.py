import numpy as np
import pandas as pd
import pytest
from src.predict import predict_dataframe
from src.train import train_model, save_model, load_model

class DummyModel:
    def predict_proba(self, x):
        return np.array([[0.8, 0.2]] * len(x))

def test_predict_adds_score_and_band():
    result = predict_dataframe(DummyModel(), pd.DataFrame({"Age":[25,31]}), ["Age"])
    assert "positive_class_probability" in result
    assert result["risk_band"].tolist() == ["Lower model score"] * 2

def test_predict_rejects_missing_features():
    with pytest.raises(ValueError, match="Missing required"):
        predict_dataframe(DummyModel(), pd.DataFrame({"Age":[25]}), ["Age","Income"])

def test_predict_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        predict_dataframe(DummyModel(), pd.DataFrame({"Age":[25]}), ["Age"], threshold=1)

def test_train_compares_models_and_persists(tmp_path):
    rng = np.random.default_rng(7)
    n = 120
    age = rng.integers(20,60,n)
    overtime = rng.choice(["Yes","No"],n)
    target = np.where((overtime == "Yes") | (age < 27), "Yes", "No")
    df = pd.DataFrame({"Age":age,"OverTime":overtime,"Attrition":target})
    result = train_model(df,"Attrition")
    assert set(result["all_metrics"]) == {"Logistic Regression","Random Forest"}
    assert 0 <= result["metrics"]["f1_score"] <= 1
    path = save_model(result,tmp_path/"model.joblib")
    restored = load_model(path)
    assert restored["feature_columns"] == result["feature_columns"]
    assert len(restored["model"].predict(df[result["feature_columns"]].head())) == 5

def test_train_rejects_single_class():
    df = pd.DataFrame({"Age":range(12),"Attrition":["No"]*12})
    with pytest.raises(ValueError, match="exactly 2 classes"):
        train_model(df,"Attrition")
