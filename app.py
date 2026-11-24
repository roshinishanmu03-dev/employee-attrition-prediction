from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.train import train_model
from src.predict import predict_dataframe
from src.data_quality import profile_dataset

st.set_page_config(page_title="PeoplePulse | HR Analytics Lab", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1450px}
[data-testid="stMetric"]{background:var(--secondary-background-color);padding:16px;border-radius:12px;border:1px solid rgba(128,128,128,.16)}
</style>""", unsafe_allow_html=True)
st.title("◉ PeoplePulse")
st.caption("Employee attrition analytics lab · Explore patterns, evaluate baseline ML models, and inspect model scores")
st.warning("Educational demonstration only. Model scores are uncertain and may reflect historical bias. Never use this application to make or automate decisions about hiring, firing, promotion, pay, or an individual employee's treatment.")

with st.sidebar:
    st.header("Data workspace")
    upload = st.file_uploader("Upload a CSV dataset", type=["csv"])
    target = st.text_input("Binary target column", "Attrition")
    st.divider()
    st.caption("Privacy first")
    st.write("Use authorized, de-identified data. Do not upload names, email addresses, phone numbers, or other direct identifiers.")
    st.caption("Expected format: one row per record and a two-class target such as Yes/No.")

if upload is None:
    st.markdown("### Welcome to your analytics workspace")
    st.write("Start with the included synthetic demonstration dataset or upload a properly authorized CSV.")
    try:
        demo = pd.read_csv("data/sample_hr_attrition.csv")
        st.success(f"Demo dataset ready · {len(demo):,} synthetic records · target: Attrition")
        if st.button("Load synthetic demo dataset", type="primary"):
            st.session_state["active_data"] = demo
            st.rerun()
    except FileNotFoundError:
        st.info("Upload a CSV to begin.")
    if "active_data" not in st.session_state:
        st.stop()
    df = st.session_state["active_data"]
else:
    try:
        df = pd.read_csv(upload)
        st.session_state["active_data"] = df
    except Exception as exc:
        st.error(f"Unable to read this CSV: {exc}")
        st.stop()

if df.empty or len(df.columns) < 2:
    st.error("Dataset must contain records and at least one feature plus a target column.")
    st.stop()
profile = profile_dataset(df)
st.markdown("### Dataset overview")
metrics = st.columns(4)
metrics[0].metric("Records", f"{profile['rows']:,}")
metrics[1].metric("Columns", profile["columns"])
metrics[2].metric("Duplicate rows", f"{profile['duplicate_rows']:,}")
metrics[3].metric("Missing cells", f"{profile['missing_percent']:.1f}%")
with st.expander("Inspect dataset and data quality"):
    st.dataframe(df.head(30), use_container_width=True)
    st.dataframe(profile["columns"], use_container_width=True, hide_index=True)
    if profile["constant_columns"]:
        st.info("Constant columns: " + ", ".join(profile["constant_columns"]))
if target not in df.columns:
    st.error(f"Target '{target}' not found. Available columns: {', '.join(map(str, df.columns))}")
    st.stop()
left, right = st.columns(2)
with left:
    st.markdown("### Target distribution")
    counts = df[target].astype("string").fillna("Missing").value_counts().rename_axis("class").reset_index(name="records")
    st.plotly_chart(px.pie(counts, names="class", values="records", hole=.55), use_container_width=True)
with right:
    st.markdown("### Train baseline models")
    st.write("Compares Logistic Regression and Random Forest using a stratified held-out test set.")
    test_size = st.slider("Test-set share", 0.1, 0.4, 0.2, 0.05)
    if st.button("Train and evaluate models", type="primary", use_container_width=True):
        with st.spinner("Training baseline models..."):
            try:
                st.session_state["training_result"] = train_model(df, target, test_size)
                st.success("Training completed.")
            except Exception as exc:
                st.error(str(exc))
result = st.session_state.get("training_result")
if result:
    st.divider()
    st.markdown("### Model evaluation")
    st.caption(f"Selected by held-out F1 score · {result['train_rows']:,} train rows · {result['test_rows']:,} test rows · Positive label: {result['positive_class']}")
    m = result["metrics"]
    cols = st.columns(5)
    for col, key, label in zip(cols, ["accuracy","balanced_accuracy","precision","recall","f1_score"],
                               ["Accuracy","Balanced accuracy","Precision","Recall","F1 score"]):
        col.metric(label, f"{m[key]:.3f}")
    st.caption(f"ROC-AUC: {m['roc_auc']:.3f} · PR-AUC: {m['pr_auc']:.3f}. Results describe this test split only.")
    compare = pd.DataFrame(result["all_metrics"]).T.reset_index(names="model")
    st.plotly_chart(px.bar(compare, x="model", y=["precision","recall","f1_score","roc_auc"],
                           barmode="group", title="Baseline model comparison", range_y=[0,1]), use_container_width=True)
    cm = result["confusion_matrix"]
    fig = go.Figure(data=go.Heatmap(z=cm, x=["Predicted 0","Predicted 1"],
        y=["Actual 0","Actual 1"], text=cm, texttemplate="%{text}", colorscale="Blues"))
    fig.update_layout(title="Confusion matrix", height=350)
    st.plotly_chart(fig, use_container_width=True)
    importance = result["importance"]
    if not importance.empty:
        st.plotly_chart(px.bar(importance.sort_values("importance"), x="importance", y="feature",
            orientation="h", title="Feature importance (descriptive, not causal)"), use_container_width=True)
    st.divider()
    st.markdown("### Batch scoring sandbox")
    st.caption("Upload a CSV containing the same feature columns. Scores are model outputs, not employee assessments.")
    threshold = st.slider("Illustrative positive-class threshold", 0.1, 0.9, 0.5, 0.05)
    prediction_upload = st.file_uploader("Upload records to score", type=["csv"], key="prediction_csv")
    if prediction_upload:
        try:
            new_data = pd.read_csv(prediction_upload)
            predictions = predict_dataframe(result["model"], new_data, result["feature_columns"],
                result["positive_class"], threshold)
            st.dataframe(predictions, use_container_width=True, hide_index=True)
            st.download_button("Download scored CSV", predictions.to_csv(index=False).encode("utf-8"),
                "peoplepulse_scored_records.csv", "text/csv")
        except Exception as exc:
            st.error(f"Scoring failed: {exc}")
else:
    st.info("Train the baseline models to unlock evaluation and batch scoring.")
st.divider()
st.caption("PeoplePulse · Portfolio learning project · Synthetic demo data only.")
