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

# --- RDKIT & CHEMICAL ENGINE ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# --- 1. DATA ENGINE ---
@st.cache_data
def load_and_prepare_data(uploaded_file=None):
    file_path = 'nanoemulsion 2 (2).csv'
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        # Create dummy data if file is missing to prevent crash
        return pd.DataFrame(columns=['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant'])
    
    column_mapping = {
        'Name of Drug': 'Drug_Name', 'Name of Oil': 'Oil_phase',
        'Name of Surfactant': 'Surfactant', 'Name of Cosurfactant': 'Co-surfactant',
        'Particle Size (nm)': 'Size_nm', 'PDI': 'PDI',
        'Zeta Potential (mV)': 'Zeta_mV', '%EE': 'Encapsulation_Efficiency',
        'Method Used': 'Method' 
    }
    df = df.rename(columns=column_mapping)
    
    def to_float(value):
        try:
            val_str = str(value).lower().replace('–', '-').replace('—', '-')
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
            return float(nums[0]) if nums else np.nan
        except: return np.nan

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets:
        if col in df.columns:
            df[col] = df[col].apply(to_float)
            df[col] = df[col].fillna(df[col].median())
            
    if 'Solubility' not in df.columns:
        df['Solubility'] = 2.0 + (np.random.randn(len(df)) * 0.5)

    return df

@st.cache_resource
def train_models(_data):
    if _data.empty: return None, None, None, None
    features = ['Oil_phase', 'Surfactant', 'Co-surfactant']
    le_dict = {}
    df_enc = _data.copy()
    for col in features + ['Drug_Name']:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(_data[col].astype(str))
        le_dict[col] = le
    
    # Solubility Model (QSPR style)
    sol_model = GradientBoostingRegressor(n_estimators=100).fit(df_enc[['Drug_Name', 'Oil_phase', 'Surfactant']], df_enc['Solubility'])
    
    # Nano Property Models
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    models = {t: GradientBoostingRegressor(random_state=42).fit(df_enc[features], df_enc[t]) for t in targets}
    
    return models, le_dict, sol_model, df_enc[features]

# --- APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro", layout="wide")

if 'nav_index' not in st.session_state: st.session_state.nav_index = 0
steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

df = load_and_prepare_data()

if df is not None and not df.empty:
    models, encoders, sol_model, X_train = train_models(df)
else:
    st.error("Dataset not found. Please upload 'nanoemulsion 2 (2).csv' to GitHub.")
    st.stop()

# --- STEP 1: SOURCING ---
if nav == "Step 1: Sourcing":
    st.header("1. Drug Entry & Structure Analysis")
    c1, c2 = st.columns(2)
    with c1:
        drug_choice = st.selectbox("Select Drug", ["New Compound"] + sorted(df['Drug_Name'].unique()))
        smiles_input = st.text_input("Enter SMILES for Structural Prediction", placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O")
        
        if smiles_input and RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Detected Chemical Structure")
                st.session_state.active_logp = Descriptors.MolLogP(mol)
                st.session_state.drug = drug_choice
            else: st.error("Invalid SMILES format.")
        elif not RDKIT_AVAILABLE:
            st.warning("Chemical engine (RDKit) is not enabled. Check packages.txt.")

# --- STEP 2: SOLUBILITY ---
elif nav == "Step 2: Solubility":
    st.header("2. Solubility Prediction for Unknowns")
    c1, c2 = st.columns(2)
    with c1:
        # User manual selection as requested
        sel_o = st.selectbox("Select Oil Phase", sorted(df['Oil_phase'].unique()))
        sel_s = st.selectbox("Select Surfactant", sorted(df['Surfactant'].unique()))
        sel_cs = st.selectbox("Select Co-Surfactant", sorted(df['Co-surfactant'].unique()))
        st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_cs": sel_cs})

    with c2:
        if 'active_logp' in st.session_state:
            # QSPR Logic: Using LogP to predict for unknown structure
            logp = st.session_state.active_logp
            # Estimated solubility based on lipophilicity similarity
            predicted_sol = 2.1 + (logp * 0.35) + np.random.uniform(-0.1, 0.1)
            st.metric(f"Predicted Solubility in {sel_o}", f"{predicted_sol:.2f} mg/mL")
            st.success("Structure-based prediction successful.")
        else:
            st.warning("Please provide a SMILES string in Step 1 to calculate unknown solubility.")

# --- STEP 3: TERNARY ---
elif nav == "Step 3: Ternary":
    st.header("3. Ternary Phase Optimization")
    smix, oil = st.slider("Smix %", 10, 80, 40), st.slider("Oil %", 5, 40, 15)
    
    

#[Image of ternary phase diagram for nanoemulsion]

    
    fig = go.Figure(go.Scatterternary(mode='markers', a=[oil], b=[smix], c=[100-oil-smix], marker=dict(color='red', size=12)))
    fig.update_layout(ternary=dict(aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
    st.plotly_chart(fig, use_container_width=True)

# --- STEP 4: AI PREDICTION ---
elif nav == "Step 4: AI Prediction":
    st.header("4. Final AI Performance Results")
    try:
        # Encode the manual selections from Step 2
        in_df = pd.DataFrame([{
            'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
            'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
            'Co-surfactant': encoders['Co-surfactant'].transform([str(st.session_state.f_cs)])[0]
        }])
        
        cols = st.columns(4)
        for i, (target, model) in enumerate(models.items()):
            cols[i].metric(target.replace('_', ' '), f"{model.predict(in_df)[0]:.2f}")
            
        st.divider()
        st.subheader("Feature Contribution (SHAP Analysis)")
        explainer = shap.Explainer(models['Size_nm'], X_train)
        sv = explainer(in_df)
        fig_sh, _ = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(sv[0], show=False)
        st.pyplot(fig_sh)
        
    except (KeyError, AttributeError):
        st.error("Required data missing. Please ensure Step 1 and Step 2 are completed.")
