import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import shap
import os
import re
import hashlib
import io
import tempfile
from fpdf import FPDF

# --- RDKIT & CHEMICAL ENGINE ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw
    import pubchempy as pcp
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# --- 1. DATA ENGINE ---
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    df = None
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            file_path = 'nanoemulsion 2 (2).csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding='latin1')

        if df is None:
            return None

        column_mapping = {
            'Name of Drug': 'Drug_Name',
            'Name of Oil': 'Oil_phase',
            'Name of Surfactant': 'Surfactant',
            'Name of Cosurfactant': 'Co-surfactant',
            'Particle Size (nm)': 'Size_nm',
            'PDI': 'PDI',
            'Zeta Potential (mV)': 'Zeta_mV',
            '%EE': 'Encapsulation_Efficiency',
            'Method Used': 'Method'
        }

        df = df.rename(columns=column_mapping)
        df.columns = [c.strip() for c in df.columns]

        def to_float(value):
            if pd.isna(value):
                return 0.0
            val_str = str(value).lower().strip()
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
            return float(nums[0]) if nums else 0.0

        for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']:
            if col in df.columns:
                df[col] = df[col].apply(to_float)

        for col in ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant', 'Method']:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df

    except:
        return None


# --- 2. MODEL TRAINING WITH EVALUATION ---
@st.cache_resource
def train_models(_data):

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

@st.cache_resource
def train_models(_data):
    if _data is None: return None, None, None, None, None
    
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    le_dict = {}
    df_enc = _data.copy()
    
    for col in features + ['Method']:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(_data[col].astype(str))
        le_dict[col] = le

    # Split data: 80% Training, 20% Testing
    X = df_enc[features]
    models = {}
    metrics_results = {}

    for t in targets:
        y = df_enc[t]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Model
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        models[t] = model
        
        # Calculate Metrics
        preds = model.predict(X_test)
        metrics_results[t] = {
            'R2': r2_score(y_test, preds),
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, preds))
        }

    # Train classification model for Method
    method_model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, df_enc['Method'])
    
    return models, le_dict, X, method_model, metrics_results

# Update the call in your main app:
# models, encoders, X_train, method_ai, model_metrics = train_models(df)


# --- 3. APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro AI", layout="wide")

df = load_and_clean_data()
models, encoders, X_train, method_ai, eval_metrics = train_models(df)

steps = ["Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps)

# --- STEP 4 ---
if nav == "Step 4: AI Prediction":

    st.header("AI Prediction & Evaluation")

    if df is not None:

        drug = st.selectbox("Drug", df['Drug_Name'].unique())
        oil = st.selectbox("Oil", df['Oil_phase'].unique())
        surf = st.selectbox("Surfactant", df['Surfactant'].unique())
        cosurf = st.selectbox("Co-Surfactant", df['Co-surfactant'].unique())

        def s_enc(col, val):
            return encoders[col].transform([val])[0]

        input_df = pd.DataFrame([{
            'Drug_Name': s_enc('Drug_Name', drug),
            'Oil_phase': s_enc('Oil_phase', oil),
            'Surfactant': s_enc('Surfactant', surf),
            'Co-surfactant': s_enc('Co-surfactant', cosurf)
        }])

        if st.button("Predict"):

            results = {t: models[t].predict(input_df)[0] for t in models}

            st.metric("Size (nm)", f"{results['Size_nm']:.2f}")
            st.metric("PDI", f"{results['PDI']:.3f}")
            st.metric("Zeta (mV)", f"{results['Zeta_mV']:.2f}")
            st.metric("EE (%)", f"{results['Encapsulation_Efficiency']:.2f}")

            st.divider()
            st.subheader("Model Evaluation Metrics (Test Data)")

            for target, metrics in eval_metrics.items():
                st.write(f"### {target}")
                st.write(f"Train Samples: {metrics['Train_Size']}")
                st.write(f"Test Samples: {metrics['Test_Size']}")
                st.write(f"R² Score: {metrics['R2']}")
                st.write(f"MAE: {metrics['MAE']}")
                st.write(f"RMSE: {metrics['RMSE']}")
                st.write("---")

            st.subheader("SHAP Explainability (Size)")
            explainer = shap.Explainer(models['Size_nm'], X_train)
            shap_values = explainer(input_df)

            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
