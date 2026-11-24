# PeoplePulse Model Card

## Summary
PeoplePulse is a portfolio learning application that compares Logistic Regression and Random Forest classifiers on a binary, tabular demonstration dataset. It produces exploratory model scores and evaluation summaries.

## Intended use
- Learning the end-to-end supervised-learning workflow
- Demonstrating preprocessing, model comparison, evaluation and batch inference
- Exploring how class imbalance affects precision, recall and F1

## Out-of-scope use
Do not use this model or its output to make, recommend, rank, or automate decisions about real employees, including hiring, firing, promotion, compensation, performance management, or access to opportunities. It is not validated for employment decisions.

## Data
The repository's demo CSV is synthetic and generated for software demonstration. It is not a representative sample of any workforce. Results on it do not establish real-world validity.

## Modeling
- Binary target column, default: `Attrition`
- Numeric features: median imputation and standard scaling
- Categorical features: most-frequent imputation and one-hot encoding
- Baselines: Logistic Regression and Random Forest
- Split: stratified train/test split with fixed random seed
- Selection: highest F1 score on the held-out split

## Evaluation
The app reports accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC and a confusion matrix. The selected model is chosen and evaluated on the same held-out split; therefore the reported score can be optimistic. Use a separate validation set, nested cross-validation, and an untouched final test set for rigorous model selection.

## Known limitations and risks
- Synthetic data is not representative of real employee populations.
- Class imbalance can make accuracy misleading; inspect recall, precision, PR-AUC and confusion matrix together.
- Historical or proxy features may encode unfair patterns.
- Feature importance is descriptive and not causal.
- Scores are probabilities from a fitted model, not certainty about an outcome.
- No fairness, calibration, subgroup performance, or deployment monitoring validation is included.

## Human oversight and privacy
Use only authorized, de-identified data for local learning. Avoid uploading names, emails, phone numbers, employee IDs, or sensitive attributes. Do not treat model outputs as evidence about any individual.
