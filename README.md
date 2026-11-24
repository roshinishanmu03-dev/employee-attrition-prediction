# PeoplePulse — Employee Attrition Analytics Lab

[![Tests](https://github.com/roshinishanmu03-dev/employee-attrition-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/roshinishanmu03-dev/employee-attrition-prediction/actions/workflows/tests.yml)

An explainable machine-learning application for exploring employee attrition patterns, comparing baseline models, and generating illustrative model-score predictions.

> **Responsible-use notice:** PeoplePulse is an educational analytics prototype. It must not be used to make or automate hiring, firing, promotion, compensation, or other employment decisions. Predictions can reflect bias in source data and are not evidence about an individual employee.

## Features

- Streamlit dashboard for HR dataset exploration
- Data-quality checks, missing-value summary and attrition distribution
- Logistic Regression and Random Forest model comparison
- Evaluation metrics: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC and PR-AUC
- Confusion matrix and grouped feature-importance view
- Batch scoring with configurable threshold and CSV export
- Synthetic sample dataset for a safe first run
- Automated tests with GitHub Actions

## Tech stack

Python · Streamlit · Pandas · NumPy · Scikit-learn · Plotly · Joblib · Pytest

## Quick start

Python 3.10+ is recommended.

```bash
git clone https://github.com/roshinishanmu03-dev/employee-attrition-prediction.git
cd employee-attrition-prediction

python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\\Scripts\\activate    # Windows

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Select **Load synthetic demo dataset**, then choose **Train and evaluate models**.

## Run tests

```bash
pytest
```

GitHub Actions runs the test suite automatically on pushes and pull requests.

## Project structure

```text
.
├── app.py
├── data/
│   ├── README.md
│   └── sample_hr_attrition.csv
├── src/
│   ├── data_quality.py
│   ├── predict.py
│   └── train.py
├── tests/
│   └── test_predict.py
├── .github/workflows/tests.yml
├── .gitignore
├── LICENSE
├── pytest.ini
└── requirements.txt
```

## Dataset

The repository includes a **synthetic demonstration dataset** with 320 records. It does not represent real employees or a real organization. For further experimentation, use only datasets you are authorized to use. Do not commit confidential employee records, personal data, or proprietary company information.

## Model limitations

- This is a learning project, not a validated employment assessment tool.
- Metrics depend on dataset quality, class balance, feature selection and train/test splitting.
- The model comparison selects the best F1 score from the same held-out split used for reporting; this can make reported performance optimistic. A separate validation set or nested cross-validation is recommended for more rigorous model selection.
- Feature importance describes model associations, not causal explanations.
- Risk bands are illustrative and must not be treated as certainty or used to make employment decisions.

## Roadmap

- [x] Add reproducible synthetic demo dataset
- [x] Add model training, evaluation and batch scoring
- [x] Add automated tests and GitHub Actions
- [ ] Verify Streamlit browser execution on the user's local machine
- [ ] Add screenshots after a successful local run
- [ ] Consider cross-validation and fairness analysis for an advanced version

## Author

**Roshini Shanmugam**  
GitHub: [@roshinishanmu03-dev](https://github.com/roshinishanmu03-dev)

## License

MIT — see [LICENSE](LICENSE).
