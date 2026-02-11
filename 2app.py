import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import shap
import os
import re

# RDKIT for SMILES analysis
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw
except ImportError:
    st.error("Missing RDKit. Please add 'rdkit' to requirements.txt")

# --- 1. DATA ENGINE (FIXED FILENAME ERROR) ---
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Search for any file matching the pattern to avoid FileNotFoundError
        possible_files = [f for f in os.listdir('.') if 'nanoemulsion' in f.lower() and f.endswith('.csv')]
        if possible_files:
            df = pd.read_csv(possible_files[0])
        else:
            return None # App will show a prompt to upload if file is missing

    column_mapping = {
        'Name of Drug': 'Drug_Name', 'Name of Oil': 'Oil_phase',
        'Name of Surfactant': 'Surfactant', 'Name of Cosurfactant': 'Co-surfactant',
        'Particle Size (nm)': 'Size_nm', 'PDI': 'PDI',
        'Zeta Potential (mV)': 'Zeta_mV', '%EE': 'Encapsulation_Efficiency'
    }
    df = df.rename(columns=column_mapping)
    df.columns = [c.strip() for c in df.columns]

    def to_float(value):
        if pd.isna(value): return np.nan
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(value))
        return float(nums[0]) if nums else np.nan

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].apply(to_float), errors='coerce').fillna(df[col].median() if not df.empty else 0)

    cat_cols = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'None'], 'Unknown')
    return df

# --- APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro", layout="wide")
if 'nav_index' not in st.session_state: st.session_state.nav_index = 0
steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

df = load_and_clean_data()

@st.cache_resource
def train_models(_data):
    if _data is None or _data.empty: return None, None, None
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    le_dict = {col: LabelEncoder().fit(_data[col].astype(str)) for col in features}
    df_enc = _data.copy()
    for col in features:
        df_enc[col] = le_dict[col].transform(_data[col].astype(str))
    
    models = {t: GradientBoostingRegressor(n_estimators=50, random_state=42).fit(df_enc[features], df_enc[t]) 
              for t in ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']}
    return models, le_dict, df_enc[features]

if df is not None:
    models, encoders, X_train = train_models(df)

# --- STEP 1: SOURCING (MODIFIED FOR SMILES) ---
if nav == "Step 1: Sourcing":
    st.header("1. Drug-Driven Sourcing")
    if df is None:
        st.error("Database not found. Please upload 'nanoemulsion 2 (2).csv'")
        up = st.file_uploader("Upload CSV", type="csv")
        if up: st.rerun()
    else:
        c1, c2 = st.columns(2)
        with c1:
            drug = st.selectbox("Database Drug", sorted(df['Drug_Name'].unique()))
            st.session_state.drug = drug
        with c2:
            smiles = st.text_input("OR Enter SMILES manually")
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    st.session_state.smiles = smiles
                    st.image(Draw.MolToImage(mol), caption="Analyzed Structure")
                else: st.error("Invalid SMILES")
        
        if st.button("Next ➡️"):
            st.session_state.nav_index = 1
            st.rerun()

# --- STEP 4: PREDICTION (SMILES DRIVEN - METHOD REMOVED) ---
elif nav == "Step 4: AI Prediction":
    st.header("4. AI Prediction Analysis")
    if 'f_o' not in st.session_state:
        st.warning("Complete previous steps first.")
    else:
        try:
            # Check if SMILES or Dropdown is being used
            use_smiles = st.session_state.get('smiles')
            
            if use_smiles:
                mol = Chem.MolFromSmiles(use_smiles)
                st.subheader("🧬 Chemical Properties Analysis")
                col_i1, col_i2 = st.columns(2)
                col_i1.metric("Molecular Weight", f"{Descriptors.MolWt(mol):.2f} Da")
                col_i2.metric("LogP (Lipophilicity)", f"{Descriptors.MolLogP(mol):.2f}")
                # We use a neutral index for manual SMILES so the model predicts based on Oil/Surf
                drug_val = 0 
            else:
                drug_val = encoders['Drug_Name'].transform([st.session_state.drug])[0]

            in_df = pd.DataFrame([{
                'Drug_Name': drug_val,
                'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
                'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
                'Co-surfactant': encoders['Co-surfactant'].transform([str(st.session_state.f_cs)])[0]
            }])

            res = {t: models[t].predict(in_df)[0] for t in models}
            
            # Display Outputs
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Size", f"{res['Size_nm']:.2f} nm")
            c2.metric("PDI", f"{res['PDI']:.3f}")
            c3.metric("Zeta", f"{res['Zeta_mV']:.2f} mV")
            c4.metric("EE %", f"{res['Encapsulation_Efficiency']:.2f}%")

            # SHAP
            st.subheader("AI Decision Logic")
            explainer = shap.Explainer(models['Size_nm'], X_train)
            sv = explainer(in_df)
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(sv[0], show=False)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
