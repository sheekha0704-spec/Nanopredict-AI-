import streamlit as st
import pandas as pd
import numpy as np
import shap

from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="NanoPredict AI", layout="wide")

st.title("NanoPredict AI - Intelligent Nanoemulsion Formulation Predictor")

# -------------------------
# Upload Dataset
# -------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset Loaded Successfully!")
else:
    st.warning("Please upload dataset to continue.")
    st.stop()

# -------------------------
# Model Training Function
# -------------------------
@st.cache_resource
def train_models(_data):

    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']

    df_enc = _data.copy()
    le_dict = {}

    # Encode categorical columns
    for col in features + ['Method']:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        le_dict[col] = le

    X = df_enc[features]

    models = {}
    evaluation_results = {}

    for target in targets:
        y = df_enc[target]

        # 80:20 Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        evaluation_results[target] = {
            "R2": round(r2, 4),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4)
        }

        models[target] = model

    # Classification Model (Method)
    method_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    method_model.fit(X, df_enc['Method'])

    return models, le_dict, X_train, method_model, evaluation_results


models, encoders, X_train, method_model, eval_metrics = train_models(df)

# -------------------------
# User Input Section
# -------------------------
st.subheader("Enter Formulation Details")

col1, col2 = st.columns(2)

with col1:
    drug = st.selectbox("Drug", df['Drug_Name'].unique())
    oil = st.selectbox("Oil Phase", df['Oil_phase'].unique())

with col2:
    surfactant = st.selectbox("Surfactant", df['Surfactant'].unique())
    cosurfactant = st.selectbox("Co-surfactant", df['Co-surfactant'].unique())

if st.button("Predict Nanoemulsion Properties"):

    input_dict = {
        'Drug_Name': encoders['Drug_Name'].transform([drug])[0],
        'Oil_phase': encoders['Oil_phase'].transform([oil])[0],
        'Surfactant': encoders['Surfactant'].transform([surfactant])[0],
        'Co-surfactant': encoders['Co-surfactant'].transform([cosurfactant])[0],
    }

    input_df = pd.DataFrame([input_dict])

    st.subheader("Predicted Properties")

    results = {}

    for target, model in models.items():
        pred = model.predict(input_df)[0]
        results[target] = round(pred, 3)

    method_pred = method_model.predict(input_df)[0]
    method_name = encoders['Method'].inverse_transform([method_pred])[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Size (nm)", results['Size_nm'])
    col2.metric("PDI", results['PDI'])
    col3.metric("Zeta Potential (mV)", results['Zeta_mV'])
    col4.metric("Encapsulation Efficiency (%)", results['Encapsulation_Efficiency'])

    st.success(f"Recommended Preparation Method: {method_name}")

    # -------------------------
    # Model Performance Section
    # -------------------------
    st.subheader("Model Performance (Test Data)")

    for target, metrics in eval_metrics.items():
        st.write(f"### {target}")
        st.write(f"R² Score: {metrics['R2']}")
        st.write(f"MAE: {metrics['MAE']}")
        st.write(f"RMSE: {metrics['RMSE']}")
        st.write("---")

    # -------------------------
    # SHAP Explainability
    # -------------------------
    st.subheader("Explainability (SHAP - Size Prediction)")

    explainer = shap.Explainer(models['Size_nm'], X_train)
    shap_values = explainer(input_df)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches='tight')
st.subheader("Model Evaluation Results")

for target, metrics in eval_metrics.items():
    st.write(f"Target: {target}")
    st.write(f"R² Score: {metrics['R2']}")
    st.write(f"MAE: {metrics['MAE']}")
    st.write(f"RMSE: {metrics['RMSE']}")
    st.write("------")
