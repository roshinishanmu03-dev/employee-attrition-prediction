# Architecture

## Data flow

```text
CSV upload / synthetic demo
          |
          v
Data profiling and target validation
          |
          v
Train/test split (stratified)
          |
          v
Preprocessing pipeline
  - numeric imputation + scaling
  - categorical imputation + one-hot encoding
          |
          v
Logistic Regression + Random Forest
          |
          v
Evaluation and baseline comparison
          |
          v
Streamlit charts and batch scoring
          |
          v
Download scored CSV
```

## Main modules

- `app.py`: Streamlit interface, data loading, charts, model training and batch scoring.
- `src/data_quality.py`: row/column counts, duplicates, missingness and constant-column profiling.
- `src/train.py`: target validation, preprocessing, training, evaluation, feature importance and model persistence helpers.
- `src/predict.py`: schema validation and batch probability scoring.
- `tests/test_predict.py`: tests for prediction validation, model training and persistence.

## Reproducibility

The training split uses a fixed random seed. The synthetic demo dataset is committed to the repository so the initial workflow can be repeated. Dependency ranges are declared in `requirements.txt`.
