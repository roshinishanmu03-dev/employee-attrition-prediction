# PeoplePulse — Employee Attrition Analytics Lab

An explainable machine-learning application for exploring employee attrition patterns, comparing baseline models, and generating illustrative attrition-risk predictions.

> **Responsible-use notice:** PeoplePulse is an educational analytics prototype. It must not be used to make or automate hiring, firing, promotion, compensation, or other employment decisions. Predictions can reflect bias in the source data and are not evidence about an individual employee.

## Features

- Streamlit dashboard for HR dataset exploration
- Data-quality checks and attrition summaries
- Logistic Regression and Random Forest model comparison
- Evaluation metrics including accuracy, precision, recall, F1 and ROC-AUC (where applicable)
- Confusion matrix and feature-importance views
- Single/batch prediction workflow and CSV export
- Automated tests and GitHub Actions test workflow
- Synthetic sample dataset for a safe first run

## Tech stack

Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly and Pytest.

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

Open the local URL printed by Streamlit. Choose the included synthetic demo data to explore the app, then train and evaluate the models.

## Run tests

```bash
pytest
```

## Project structure

```text
.
├── app.py
├── data/
│   ├── README.md
│   └── sample_hr_attrition.csv
├── docs/
├── models/
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

The repository includes a synthetic demonstration dataset. If using the IBM HR Analytics Attrition dataset for further experimentation, obtain it from its authorized source and review its terms. Do not commit confidential employee records, personal data, or proprietary company information.

## Model limitations

- This is a learning project, not a validated employment assessment tool.
- Metrics depend on dataset quality, class balance, feature selection and train/test splitting.
- Feature importance describes model associations, not causal explanations.
- Risk bands are illustrative and should not be interpreted as certainty.

## Roadmap

- [ ] Verify installation and browser execution on a clean environment
- [ ] Run tests in GitHub Actions
- [ ] Add screenshots after a successful local run
- [ ] Document model results from a reproducible evaluation

## Author

**Roshini Shanmugam**  
GitHub: [@roshinishanmu03-dev](https://github.com/roshinishanmu03-dev)

## License

MIT — see [LICENSE](LICENSE).
