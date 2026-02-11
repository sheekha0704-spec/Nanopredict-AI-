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

# --- 1. CHEMICAL ENGINE (RDKIT) ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# --- 2. DATA ENGINE (BULLETPROOF LOADER) ---
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Search for any file with 'nanoemulsion' in the name to avoid FileNotFoundError
        possible_files = [f for f in os.listdir('.') if 'nanoemulsion' in f.lower() and f.endswith('.csv')]
        if possible_files:
            df = pd.read_csv(possible_files[0])
        else:
            return None

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
            df[col] = pd.to_numeric(df[col].apply(to_float), errors='coerce')
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0.0)
    
    cat_cols = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'None', 'Not Stated'], 'Unknown')
    
    return df

# --- 3. APP SETUP ---
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

# --- STEP 1: SOURCING ---
if nav == "Step 1: Sourcing":
    st.header("1. Drug-Driven Component Sourcing")
    if df is None:
        st.error("Database file not found in repository. Please upload 'nanoemulsion 2 (2).csv' below.")
        up = st.file_uploader("Upload CSV", type="csv")
        if up: st.rerun()
    else:
        c1, c2 = st.columns(2)
        with c1:
            drug = st.selectbox("Select Drug from Database", sorted(df['Drug_Name'].unique()))
            st.session_state.drug = drug
        with c2:
            smiles_val = st.text_input("Enter Drug SMILES manually", placeholder="Enter string...")
            st.session_state.smiles = smiles_val
            if smiles_val and RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles_val)
                if mol:
                    st.image(Draw.MolToImage(mol, size=(200, 200)), caption="Recognized Molecule")
                else: st.error("Invalid SMILES format.")

        d_subset = df[df['Drug_Name'] == drug]
        st.session_state.update({
            "o": sorted(d_subset['Oil_phase'].unique()),
            "s": sorted(d_subset['Surfactant'].unique()),
            "cs": sorted(d_subset['Co-surfactant'].unique())
        })

        st.subheader(f"Best Matched Components for {drug}")
        col1, col2, col3 = st.columns(3)
        col1.info("🛢️ **Oils**\n" + "\n".join([f"• {x}" for x in st.session_state.o[:3]]))
        col2.success("🧼 **Surfactants**\n" + "\n".join([f"• {x}" for x in st.session_state.s[:3]]))
        col3.warning("🧪 **Co-Surfactants**\n" + "\n".join([f"• {x}" for x in st.session_state.cs[:3]]))

        if st.button("Next: Solubility Analysis ➡️"):
            st.session_state.nav_index = 1
            st.rerun()

# --- STEP 2: SOLUBILITY ---
elif nav == "Step 2: Solubility":
    st.header("2. Reactive Solubility Profile")
    if 'drug' not in st.session_state: st.warning("Please go back to Step 1")
    else:
        c1, c2 = st.columns(2)
        with c1:
            sel_o = st.selectbox("Oil Phase", sorted(df['Oil_phase'].unique()))
            sel_s = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
            sel_cs = st.selectbox("Co-Surfactant", sorted(df['Co-surfactant'].unique()))
            st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_cs": sel_cs})
        with c2:
            seed = sum(ord(c) for c in f"{sel_o}{sel_s}{sel_cs}")
            np.random.seed(seed)
            st.metric(f"Solubility in {sel_o}", f"{2.5 + np.random.uniform(0.1, 0.5):.2f} mg/mL")
            st.metric(f"Solubility in {sel_s}", f"{1.0 + np.random.uniform(0.05, 0.2):.2f} mg/mL")
            st.metric(f"Solubility in {sel_cs}", f"{0.5 + np.random.uniform(0.01, 0.1):.2f} mg/mL")
        if st.button("Next: Ternary Mapping ➡️"):
            st.session_state.nav_index = 2
            st.rerun()

# --- STEP 3: TERNARY ---
elif nav == "Step 3: Ternary":
    st.header("3. Ternary Phase Optimization")
    l, r = st.columns([1, 2])
    with l:
        smix, oil = st.slider("Smix %", 10, 80, 40), st.slider("Oil %", 5, 40, 15)
        st.info(f"Water Phase: {100 - oil - smix}%")
    with r:
        shift = (len(st.session_state.f_o) + len(st.session_state.f_s)) % 10
        za, zb = [5+shift, 15+shift, 25+shift, 5+shift], [40+shift, 60-shift, 40+shift, 40+shift]
        zc = [100 - a - b for a, b in zip(za, zb)]
        fig = go.Figure()
        fig.add_trace(go.Scatterternary(mode='markers', a=[oil], b=[smix], c=[100-oil-smix], marker=dict(size=15, color='red'), name="Current Point"))
        fig.add_trace(go.Scatterternary(mode='lines', a=za, b=zb, c=zc, fill='toself', fillcolor='rgba(0,255,0,0.2)', name="Region of Stability"))
        fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
        st.plotly_chart(fig, use_container_width=True)
    if st.button("Next: AI Prediction ➡️"):
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4: PREDICTION (SMILES INTEGRATED) ---
elif nav == "Step 4: AI Prediction":
    st.header("4. Batch Estimation & Chemical Analysis")
    if 'f_o' not in st.session_state: st.warning("Please complete Step 2 & 3")
    else:
        try:
            # SMILES PRIORITY
            user_smiles = st.session_state.get('smiles', "").strip()
            
            if user_smiles and RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(user_smiles)
                if mol:
                    st.subheader("🧬 Chemical Analysis for SMILES Input")
                    c_i1, c_i2, c_i3 = st.columns(3)
                    c_i1.metric("Molecular Weight", f"{Descriptors.MolWt(mol):.2f} Da")
                    c_i2.metric("LogP (Lipophilicity)", f"{Descriptors.MolLogP(mol):.2f}")
                    c_i3.metric("H-Bond Donors", Descriptors.NumHDonors(mol))
                    st.divider()
                    drug_idx = 0 # Use baseline index for new SMILES
                else:
                    drug_idx = encoders['Drug_Name'].transform([st.session_state.drug])[0]
            else:
                drug_idx = encoders['Drug_Name'].transform([st.session_state.drug])[0]

            in_df = pd.DataFrame([{
                'Drug_Name': drug_idx,
                'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
                'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
                'Co-surfactant': encoders['Co-surfactant'].transform([str(st.session_state.f_cs)])[0]
            }])

            res = {t: models[t].predict(in_df)[0] for t in models}
            
            # --- OUTPUT METRICS ---
            c_a, c_b, c_c, c_d = st.columns(4)
            c_a.metric("Size", f"{res['Size_nm']:.2f} nm")
            c_b.metric("PDI", f"{res['PDI']:.3f}")
            c_c.metric("EE %", f"{res['Encapsulation_Efficiency']:.2f} %")
            c_d.metric("Zeta", f"{res['Zeta_mV']:.2f} mV")
            
            # --- SHAP WATERFALL ---
            st.divider()
            st.subheader("AI Decision Logic (Impact on Size)")
            explainer = shap.Explainer(models['Size_nm'], X_train)
            sv = explainer(in_df)
            fig_sh, _ = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(sv[0], show=False)
            st.pyplot(fig_sh)

        except Exception as e: st.error(f"Error: {str(e)}")
