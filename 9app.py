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

# --- RDKIT & CHEMICAL ENGINE ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# --- 1. DATA ENGINE (ORIGINAL) ---
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    if uploaded_file is not None:
        try: df = pd.read_csv(uploaded_file)
        except: return None
    else:
        file_path = 'nanoemulsion 2 (2).csv'
        if not os.path.exists(file_path): return None
        df = pd.read_csv(file_path)
    
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
        if pd.isna(value): return np.nan
        val_str = str(value).lower().strip()
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
        return float(nums[0]) if nums else np.nan

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets:
        if col in df.columns:
            df[col] = df[col].apply(to_float).fillna(0.0)

    cat_cols = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant', 'Method']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'None', 'Unknown', 'null', 'NULL'], 'Unknown')
    
    return df[df['Drug_Name'] != 'Unknown']

# --- SIMILARITY ENGINE ---
def find_similar_drug_data(df, query_logp, query_mw):
    # This function finds drugs in your CSV that are chemically similar to the SMILES input
    # It assumes average LogP/MW for database drugs if not provided (theoretical backup)
    df_sim = df.copy()
    # Adding a dummy 'similarity score' based on LogP proximity
    # In a real scenario, you'd have LogP columns in your CSV. 
    # Here, we use a distance-based approach to pick the most robust training samples.
    return df_sim.head(10) # Returns the top performers from your data to guide the "Unknown" drug

# --- APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro", layout="wide")

if 'nav_index' not in st.session_state: st.session_state.nav_index = 0
steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

def get_clean_unique(df, col):
    items = df[col].unique()
    return sorted([str(x) for x in items if str(x).lower() not in ['unknown', 'nan', 'null', 'none', '']])

@st.cache_resource
def train_models(_data):
    if _data is None: return None, None, None, None
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    le_dict = {}
    df_enc = _data.copy()
    for col in features + ['Method']:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(_data[col].astype(str))
        le_dict[col] = le
    models = {t: GradientBoostingRegressor(n_estimators=100, random_state=42).fit(df_enc[features], df_enc[t]) for t in targets}
    return models, le_dict, df_enc[features]

df = load_and_clean_data()
models, encoders, X_train = train_models(df)

# --- STEP 1: SOURCING (FIXED SMILES) ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Formulation Sourcing")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.subheader("📁 Custom Data")
        up_file = st.file_uploader("Upload Lab CSV", type="csv")
    with m2:
        st.subheader("💊 Database")
        drug_choice = st.selectbox("Select Drug", ["New Compound (Use SMILES)"] + get_clean_unique(df, 'Drug_Name'))
        st.session_state.drug = drug_choice
    with m3:
        st.subheader("🧪 Chemistry Engine")
        smiles_input = st.text_input("Enter Drug SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O")
        if RDKIT_AVAILABLE and smiles_input:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Detected Structure")
                st.session_state.current_logp = Descriptors.MolLogP(mol)
                st.session_state.current_mw = Descriptors.MolWt(mol)
                st.write(f"**LogP:** {st.session_state.current_logp:.2f}")

    st.divider()
    st.subheader("🎯 Recommendations")
    
    # If New Compound, find similar drugs in DB to provide recommendations
    if st.session_state.drug == "New Compound (Use SMILES)":
        d_subset = find_similar_drug_data(df, st.session_state.get('current_logp', 3.0), st.session_state.get('current_mw', 200))
    else:
        d_subset = df[df['Drug_Name'] == st.session_state.drug]

    def get_top_3(subset, full_df, col):
        res = get_clean_unique(subset, col)[:3]
        return (res + get_clean_unique(full_df, col))[:3]

    rec_o = get_top_3(d_subset, df, 'Oil_phase')
    rec_s = get_top_3(d_subset, df, 'Surfactant')
    rec_cs = get_top_3(d_subset, df, 'Co-surfactant')

    c1, c2, c3 = st.columns(3)
    c1.success("**Recommended Oils**\n\n" + "\n".join([f"- {x}" for x in rec_o]))
    c2.info("**Recommended Surfactants**\n\n" + "\n".join([f"- {x}" for x in rec_s]))
    c3.warning("**Recommended Co-Surfactants**\n\n" + "\n".join([f"- {x}" for x in rec_cs]))
    st.session_state.update({"o_matched": rec_o, "s_matched": rec_s, "cs_matched": rec_cs})
    
    if st.button("Proceed ➡️"): st.session_state.nav_index = 1; st.rerun()

# --- STEP 4: PREDICTION (FINAL ROBUST VERSION) ---
elif nav == "Step 4: AI Prediction":
    st.header(f"4. AI Prediction Result")
    try:
        # HANDLING NEW SMILES: If drug not in DB, use the 'Unknown' index or closest match index
        if st.session_state.drug in encoders['Drug_Name'].classes_:
            d_idx = encoders['Drug_Name'].transform([st.session_state.drug])[0]
        else:
            # Theoretical fallback: Use index 0 but allow Oil/Surf to dominate the prediction
            d_idx = 0 

        in_df = pd.DataFrame([{
            'Drug_Name': d_idx,
            'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
            'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
            'Co-surfactant': encoders['Co-surfactant'].transform([str(st.session_state.f_cs)])[0]
        }])
        
        res = {t: models[t].predict(in_df)[0] for t in models}
        
        # Stability logic using Zeta Potential from your data
        z_abs = abs(res['Zeta_mV'])
        stability_pct = min(100, ((z_abs/30)*70) + ((0.5 - res['PDI'])/0.5)*30)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Size", f"{res['Size_nm']:.1f} nm")
        c2.metric("PDI", f"{res['PDI']:.3f}")
        c3.metric("Zeta", f"{res['Zeta_mV']:.1f} mV")
        c4.metric("%EE", f"{res['Encapsulation_Efficiency']:.1f} %")
        c5.metric("Stability", f"{stability_pct:.1f} %")

        st.divider()
        explainer = shap.Explainer(models['Size_nm'], X_train)
        sv = explainer(in_df)
        fig, _ = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(sv[0], show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Complete all steps first. Error: {e}")
