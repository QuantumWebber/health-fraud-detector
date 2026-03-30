import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load the trained model
lgbm = joblib.load('../data/lgbm_model.pkl')

# Page config
st.set_page_config(page_title="Health Insurance Fraud Detector", page_icon="🏥")

# Title
st.title("🏥 Health Insurance Fraud Detector")
st.write("Enter provider details to check if the claim is fraudulent")
st.markdown("---")

# Input fields
prov_count = st.number_input("How many claims has this provider made?", 0, 5000, 100)
prov_total = st.number_input("Provider total billing amount (Rs)", 0, 10000000, 500000)
prov_avg   = st.number_input("Provider average claim amount (Rs)", 0, 100000, 5000)
prov_bene  = st.number_input("Provider unique patients count", 0, 5000, 80)
claim_amt  = st.number_input("This claim amount (Rs)", 0, 500000, 5000)
age        = st.slider("Patient age", 0, 100, 65)
chronic    = st.slider("Number of chronic conditions", 0, 11, 2)

st.markdown("---")

# Predict button
if st.button("🔍 Check Fraud", type="primary"):

    # Build input dataframe
    input_data = pd.DataFrame([[
        claim_amt, 1000, age, 1, chronic,
        prov_count, prov_avg, prov_total, prov_bene,
        12, 12, 0, 0, 10000, 1000, 1, 1, 0
    ]], columns=[
        'InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'Age',
        'ClaimDurationDays', 'chronic_score', 'provider_claim_count',
        'provider_avg_claim', 'provider_total_amt', 'provider_unique_bene',
        'NoOfMonths_PartACov', 'NoOfMonths_PartBCov',
        'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
        'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt',
        'Gender', 'Race', 'ClaimType'
    ])

    # Get prediction
    prob = lgbm.predict_proba(input_data)[0][1]
    pred = lgbm.predict(input_data)[0]

    st.markdown("---")

    # Show result
    if pred == 1:
        st.error(f"🚨 FRAUD DETECTED!")
        st.error(f"Fraud probability: {prob*100:.1f}%")
    else:
        st.success(f"✅ LEGITIMATE CLAIM")
        st.success(f"Fraud probability: {prob*100:.1f}%")

    # Fraud risk bar
    st.write("**Fraud Risk Level:**")
    st.progress(float(prob))

    st.markdown("---")

    # SHAP explanation
    st.write("### 🔍 Why did the model give this result?")
    st.write("This chart shows which factors pushed the prediction toward fraud (red) or away from fraud (blue).")

    try:  # ✅ Fixed: was missing proper indentation (4 spaces inside the if block)
        explainer = shap.TreeExplainer(lgbm)
        shap_vals = explainer.shap_values(input_data)

        # Handle both old and new shap output formats
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
            bv = float(explainer.expected_value[1])
        else:
            sv = shap_vals[0]
            bv = float(explainer.expected_value)

        # Feature names and values
        feature_names = input_data.columns.tolist()
        feature_values = input_data.iloc[0].values

        # Sort by absolute SHAP value
        import numpy as np
        indices = np.argsort(np.abs(sv))[::-1][:10]  # top 10 features

        top_names  = [feature_names[i] for i in indices]
        top_vals   = [sv[i] for i in indices]
        top_data   = [feature_values[i] for i in indices]

        # Draw bar chart manually — no shap.plots
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in top_vals]

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')

        bars = ax.barh(
            [f"{n}\n(value={v:.0f})" for n, v in zip(top_names, top_data)],
            top_vals,
            color=colors
        )

        ax.axvline(0, color='white', linewidth=0.8)
        ax.set_xlabel("SHAP value (red = pushes toward fraud, blue = away from fraud)",
                      color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.warning(f"SHAP chart could not load: {e}")