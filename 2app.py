import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import shap
import os
import re

# RDKIT IMPORTS
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from PIL import Image

# --- 1. DATA ENGINE ---
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    # Try to find the file even if there are hidden spaces in the name
    file_name = 'nanoemulsion 2 (2).csv'
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        # Emergency check for similar filenames in directory
        files = [f for f in os.listdir('.') if 'nanoemulsion' in f.lower()]
        if files:
            df = pd.read_csv(files[0])
        else:
            return None
    
    column_mapping = {
        'Name of Drug': 'Drug_Name', 'Name of Oil': 'Oil_phase',
        'Name of Surfactant': 'Surfactant', 'Name of Cosurfactant': 'Co-surfactant',
        'Particle Size (nm)': 'Size_nm', 'PDI': 'PDI',
        'Zeta Potential (mV)': 'Zeta_mV', '%EE': 'Encapsulation_Efficiency',
        'Method Used': 'Method' 
    }
    df = df.rename(columns=column_mapping)
    df.columns = [c.strip() for c in df.columns]

    def to_float(value):
        if pd.isna(value): return 0.0
        val_str = str(value).lower().strip()
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
        return float(nums[0]) if nums else 0.0

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets:
        if col in df.columns:
            df[col] = df[col].apply(to_float).fillna(0.0)

    cat_cols = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant', 'Method']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'None'], 'Unknown')
        else:
            df[col] = 'Unknown'
    return df

# --- CHEMICAL UTILS ---
def get_mol_data(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return mol, {'MW': Descriptors.MolWt(mol), 'LogP': Descriptors.MolLogP(mol)}
    except: pass
    return None, None

# --- APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro", layout="wide")
if 'nav_index' not in st.session_state: st.session_state.nav_index = 0
steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

df = load_and_clean_data()

@st.cache_resource
def train_models(_data):
    if _data is None or _data.empty: return None, None, None, None
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    le_dict = {}
    df_enc = _data.copy()
    for col in features + ['Method']:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(_data[col].astype(str))
        le_dict[col] = le
    
    models = {t: GradientBoostingRegressor(n_estimators=50).fit(df_enc[features], df_enc[t]) for t in ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']}
    method_model = RandomForestClassifier(n_estimators=50).fit(df_enc[features], df_enc['Method'])
    return models, le_dict, df_enc[features], method_model

if df is not None:
    models, encoders, X_train, method_ai = train_models(df)

# --- STEP 1 ---
if nav == "Step 1: Sourcing":
    st.header("1. Sourcing")
    if df is None:
        st.error("CSV File not found. Please upload it below.")
        up = st.file_uploader("Upload CSV", type="csv")
        if up: st.rerun()
    else:
        c1, c2 = st.columns(2)
        with c1:
            drug = st.selectbox("Select Drug", sorted(df['Drug_Name'].unique()))
            st.session_state.drug = drug
        with c2:
            st.session_state.smiles = st.text_input("Enter SMILES", value=st.session_state.get('smiles', ""))
        if st.button("Next ➡️"):
            st.session_state.nav_index = 1
            st.rerun()

# --- STEP 2 ---
elif nav == "Step 2: Solubility":
    st.header("2. Solubility")
    if 'drug' not in st.session_state: st.warning("Go back to Step 1")
    else:
        st.session_state.f_o = st.selectbox("Oil", sorted(df['Oil_phase'].unique()))
        st.session_state.f_s = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
        st.session_state.f_cs = st.selectbox("Co-Surfactant", sorted(df['Co-surfactant'].unique()))
        if st.button("Next ➡️"):
            st.session_state.nav_index = 2
            st.rerun()

# --- STEP 3 ---
elif nav == "Step 3: Ternary":
    st.header("3. Ternary Optimization")
    oil = st.slider("Oil %", 5, 40, 15)
    smix = st.slider("Smix %", 10, 80, 40)
    fig = go.Figure(go.Scatterternary(a=[oil], b=[smix], c=[100-oil-smix]))
    st.plotly_chart(fig)
    if st.button("Next ➡️"):
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4 ---
elif nav == "Step 4: AI Prediction":
    st.header("4. Prediction & Chemical Intel")
    if 'f_o' not in st.session_state: st.error("Incomplete steps")
    else:
        try:
            # Handle SMILES
            smiles = st.session_state.get('smiles', "").strip()
            if smiles:
                mol, info = get_mol_data(smiles)
                if mol:
                    st.image(Draw.MolToImage(mol, size=(250, 250)), caption="Molecular Structure")
                    st.success(f"MW: {info['MW']:.1f} | LogP: {info['LogP']:.1f}")

            # Prepare Input
            try: d_idx = encoders['Drug_Name'].transform([st.session_state.drug])[0]
            except: d_idx = 0
            
            in_df = pd.DataFrame([{
                'Drug_Name': d_idx,
                'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
                'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
                'Co-surfactant': encoders['Co-surfactant'].transform([st.session_state.f_cs])[0]
            }])

            # Predict
            res = {t: models[t].predict(in_df)[0] for t in models}
            meth = encoders['Method'].inverse_transform([method_ai.predict(in_df)[0]])[0]

            c1, c2 = st.columns(2)
            c1.metric("Size", f"{res['Size_nm']:.2f} nm")
            c1.metric("PDI", f"{res['PDI']:.3f}")
            c2.metric("EE %", f"{res['Encapsulation_Efficiency']:.1f}%")
            c2.success(f"Method: {meth}")

            # SHAP
            st.divider()
            explainer = shap.Explainer(models['Size_nm'], X_train)
            sv = explainer(in_df)
            fig, ax = plt.subplots(figsize=(8,3))
            shap.plots.bar(sv[0], show=False)
            st.pyplot(fig)

        except Exception as e: st.error(f"Error: {e}")
