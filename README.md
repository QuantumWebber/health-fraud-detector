# 🏥 Healthcare Insurance Fraud Detection

> **Detecting fraudulent Medicare providers using machine learning on real CMS claims data.**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-Fast%20Training-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Problem Statement

Healthcare insurance fraud costs the U.S. government **billions of dollars annually**. Fraudulent providers submit inflated, duplicate, or fabricated claims to Medicare — making automated detection critical for financial and public health systems.

This project builds a **binary classification pipeline** to identify whether a Medicare provider is **potentially fraudulent**, using real-world CMS (Centers for Medicare & Medicaid Services) claims data.

---

## 📂 Dataset

The dataset is sourced from Kaggle: [Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)

| File | Description |
|------|-------------|
| `Train-1542865627584.csv` | Provider fraud labels (`PotentialFraud`: Yes/No) |
| `Train_Beneficiarydata-*.csv` | Patient demographics, chronic conditions, coverage info |
| `Train_Inpatientdata-*.csv` | Inpatient hospitalization claims |
| `Train_Outpatientdata-*.csv` | Outpatient visit claims |

All four files are merged at the **Provider** and **BeneID** level to create one unified modeling dataset.

---

## 🗂️ Project Structure

```
healthcare-fraud-detection/
│
├── data/
│   ├── Train-1542865627584.csv
│   ├── Train_Beneficiarydata-1542865627584.csv
│   ├── Train_Inpatientdata-1542865627584.csv
│   ├── Train_Outpatientdata-1542865627584.csv
│   ├── merged_data.csv          # EDA output
│   ├── X_features.csv           # Feature matrix
│   ├── y_target.csv             # Target labels
│   ├── X_train.pkl / X_test.pkl
│   ├── y_train.pkl / y_test.pkl
│   ├── xgb_model.pkl
│   └── lgbm_model.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Feature creation & preprocessing
│   └── 03_modelling.ipynb       # Model training, evaluation & SHAP
│
└── README.md
```

---

## 🔍 Approach

### 1. Exploratory Data Analysis (`01_eda.ipynb`)
- Loaded and merged 4 CSV files (fraud labels, beneficiary, inpatient, outpatient claims)
- Analyzed fraud distribution — class imbalance identified (~10% fraud providers)
- Visualized claim amount distributions: **fraud providers have significantly higher average claim amounts**
- Analyzed chronic condition rates across fraud vs. non-fraud providers
- Computed provider-level stats (total claims, average claim amount, total reimbursements)

### 2. Feature Engineering (`02_feature_engineering.ipynb`)
Engineered the following features from raw data:

| Feature | Description |
|---------|-------------|
| `Age` | Derived from DOB relative to Dec 2009 |
| `ClaimDurationDays` | Inpatient stay duration in days |
| `chronic_score` | Sum of all 11 chronic condition flags |
| `provider_claim_count` | Total claims filed by a provider |
| `provider_avg_claim` | Average reimbursement amount per provider |
| `provider_total_amt` | Total reimbursements billed by provider |
| `provider_unique_bene` | Unique beneficiaries per provider |
| `IPAnnualReimbursementAmt` | Annual inpatient reimbursement |
| `OPAnnualReimbursementAmt` | Annual outpatient reimbursement |
| `Gender`, `Race`, `ClaimType` | Label-encoded categorical features |

- Class imbalance handled using **`scale_pos_weight`** (XGBoost) and **`class_weight='balanced'`** (LightGBM)
- No SMOTE needed — tree-based models handle imbalance natively

### 3. Modelling (`03_modelling.ipynb`)
Two gradient boosting models were trained and compared:

#### ✅ XGBoost (with GridSearchCV)
- Hyperparameter tuning over: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- 3-fold cross-validation with `roc_auc` as scoring metric

#### ✅ LightGBM
- Faster training, better performance on imbalanced datasets
- Final model selected for business impact analysis

---

## 📊 Results

| Model | AUC-ROC |
|-------|---------|
| XGBoost (tuned) | ~0.87+ |
| LightGBM | ~0.88+ |

> *Exact scores depend on your run; AUC-ROC is the primary metric for fraud detection (handles class imbalance better than accuracy)*

### 💼 Business Impact (LightGBM)

```
Fraud claims caught:     [TP value]
Fraud claims missed:     [FN value]
Estimated money saved:   Rs [TP × 15,000]
Detection rate:          ~XX%
```

---

## 📈 Model Explainability — SHAP

SHAP (SHapley Additive exPlanations) was used to interpret the LightGBM model:

- **`provider_total_amt`** and **`provider_claim_count`** are top fraud predictors
- Providers with **higher claim volumes and amounts** are significantly more likely to be fraudulent
- **`chronic_score`** and **`ClaimDurationDays`** also contribute meaningfully

SHAP summary plot gives a global view of feature importance with directionality — essential for explaining model decisions to stakeholders.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | XGBoost, LightGBM, Scikit-learn |
| Explainability | SHAP |
| Model Persistence | Joblib |

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/QuantumWebber/healthcare-fraud-detection.git
cd healthcare-fraud-detection

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap joblib

# Download dataset from Kaggle and place in /data

# Run notebooks in order
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_modelling.ipynb
```

---

## 🧠 Key Learnings

- Real-world healthcare data requires **multi-table joins** — merging provider, beneficiary, and claims data is non-trivial
- **AUC-ROC** is a far better metric than accuracy for imbalanced fraud datasets
- **Provider-level aggregations** (total claims, unique patients) are more powerful fraud signals than individual claim features
- **SHAP** makes black-box models explainable — critical for healthcare/finance use cases

---

## 👤 Author

**Jatin** — B.Tech Engineering Physics, Delhi Technological University  
GitHub: [@QuantumWebber](https://github.com/QuantumWebber)

---

