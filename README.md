# Health Insurance Fraud Detector

A Streamlit app that scores whether a health insurance claim looks fraudulent, using a pretrained LightGBM model plus a live SHAP explanation of *why* it made that call.

---

## What This Repo Actually Contains

This repo holds the **inference app only** — a Streamlit UI wrapping a pre-trained, pre-saved LightGBM model (`data/lgbm_model.pkl`). The training pipeline (EDA, feature engineering, model comparison against XGBoost, CMS Medicare dataset joins) that produced this model is **not included in this repo** — only the resulting `.pkl` and the app that consumes it are. Be accurate about that distinction in any writeup: this repo is a deployment/demo artifact, not the modeling project itself.

## What The App Does

- Takes manually entered provider- and claim-level inputs (provider claim count, provider total/average billing, unique patient count, this claim's amount, patient age, chronic condition count) via Streamlit input widgets.
- Builds a single-row feature dataframe matching the 18 features the model expects (some — like `NoOfMonths_PartACov`, deductible amounts — are hardcoded placeholder values in the current app, not user inputs; worth flagging if you demo this live).
- Runs `lgbm.predict_proba()` to get a fraud probability and `lgbm.predict()` for the binary flag, and displays a risk verdict with a progress bar.
- Explains the prediction with `shap.TreeExplainer`, plotting the top 10 features by absolute SHAP value as a custom red/blue horizontal bar chart (red = pushes toward fraud, blue = pushes away) — handles both list- and array-style SHAP output formats defensively.

---

## Tech Stack

Streamlit, LightGBM, SHAP, Pandas, Matplotlib, Joblib

---

## Project Structure (as it exists in this repo)

```
health-fraud-detector/
├── app/
│   └── app.py              # Streamlit UI + inference + SHAP explanation
├── data/
│   └── lgbm_model.pkl      # pretrained LightGBM model (training pipeline not in this repo)
└── requirements.txt
```

---


## How to Run

```bash
pip install -r requirements.txt
cd app
streamlit run app.py
```
