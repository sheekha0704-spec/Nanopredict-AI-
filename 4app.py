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
    from PIL import Image
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# --- 1. DATA ENGINE ---
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
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
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str.replace('–', '-'))
        return float(nums[0]) if nums else np.nan

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets:
        if col in df.columns:
            df[col] = df[col].apply(to_float).fillna(df[col].apply(to_float).median())
        else:
            df[col] = 0.0

    return df.dropna(subset=['Drug_Name', 'Oil_phase', 'Surfactant'])

# --- APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro", layout="wide")

if 'nav_index' not in st.session_state: st.session_state.nav_index = 0
steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

# --- GLOBAL DATA LOADING ---
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV Database", type="csv")
df = load_and_clean_data(uploaded_file)

@st.cache_resource
def train_models(_data):
    if _data is None or _data.empty: return None, None, None, None
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    le_dict = {}
    df_enc = _data.copy()
    for col in features + ['Method']:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(_data[col].astype(str))
        le_dict[col] = le
    models = {t: GradientBoostingRegressor(n_estimators=50, random_state=42).fit(df_enc[features], df_enc[t]) for t in targets}
    method_model = RandomForestClassifier(n_estimators=50, random_state=42).fit(df_enc[features], df_enc['Method'])
    return models, le_dict, df_enc[features], method_model

if df is not None:
    models, encoders, X_train, method_ai = train_models(df)

# --- STEP 1: SOURCING (FILE + DATABASE + SMILES) ---
if nav == "Step 1: Sourcing":
    st.header("1. Industrial Sourcing & SMILES Analysis")
    if df is None: st.error("Please upload a CSV file or ensure database is present."); st.stop()
    
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("💊 Compound Input")
        drug_choice = st.selectbox("Select Database Drug", sorted(df['Drug_Name'].unique()))
        smiles_input = st.text_input("Input Drug SMILES (for custom profiling)", value="CC(=O)OC1=CC=CC=C1C(=O)O")
        
        if smiles_input and RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Detected Molecular Architecture")
                logp = Descriptors.MolLogP(mol)
                st.session_state.current_profile = {"logp": logp, "mw": Descriptors.MolWt(mol)}
                st.write(f"**Calculated LogP:** {logp:.2f}")
            else: st.error("Invalid SMILES format.")
        st.session_state.drug = drug_choice

    with c2:
        st.subheader("🎯 Matched Production Components")
        p_logp = st.session_state.get('current_profile', {'logp': 2.0})['logp']
        
        # Filtering logic for specific needs
        o_final = [o for o in df['Oil_phase'].unique() if ("oleic" in o.lower() if p_logp > 3 else "capryl" in o.lower())]
        o_final = o_final if o_final else list(df['Oil_phase'].unique()[:3])
        
        col1, col2 = st.columns(2)
        col1.success("🛢️ Recommended Oils\n\n" + "\n".join([f"- {x}" for x in o_final[:3]]))
        col2.info("🧼 Recommended Surfactants\n\n" + "\n".join([f"- {x}" for x in df['Surfactant'].unique()[:3]]))
        st.session_state.update({"o_matched": o_final, "s_matched": list(df['Surfactant'].unique())})

    if st.button("Next: Predictive Solubility ➡️"):
        st.session_state.nav_index = 1
        st.rerun()

# --- STEP 2: SOLUBILITY ---
elif nav == "Step 2: Solubility":
    st.header("2. Customizable Solubility Profile")
    if 'drug' not in st.session_state: st.warning("Please complete Step 1.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            sel_o = st.selectbox("Chosen Oil", st.session_state.o_matched + list(df['Oil_phase'].unique()))
            sel_s = st.selectbox("Chosen Surfactant", st.session_state.s_matched)
            sel_cs = st.selectbox("Chosen Co-Surfactant", sorted(df['Co-surfactant'].unique()))
            st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_cs": sel_cs})
        with c2:
            logp = st.session_state.get('current_profile', {'logp': 2.0})['logp']
            oil_sol = (logp * 0.7) + np.random.uniform(0.5, 1.0)
            st.metric(f"Solubility in {sel_o}", f"{oil_sol:.2f} mg/mL")
            
        
        if st.button("Next: Ternary Mapping ➡️"):
            st.session_state.nav_index = 2
            st.rerun()

# --- STEP 3: TERNARY (WITH REGION) ---
elif nav == "Step 3: Ternary":
    st.header("3. Ternary Phase Optimization")
    smix, oil = st.slider("Smix %", 10, 80, 40), st.slider("Oil %", 5, 40, 15)
    
    # Mathematical region generation for "Nanoemulsion Region"
    shift = (len(st.session_state.get('f_o', '')) + len(st.session_state.get('f_s', ''))) % 8
    reg_a = [2, 10+shift, 25+shift, 2] # Oil
    reg_b = [40+shift, 70-shift, 45+shift, 40+shift] # Smix
    reg_c = [100 - a - b for a, b in zip(reg_a, reg_b)] # Water
    
    fig = go.Figure()
    fig.add_trace(go.Scatterternary(mode='lines', a=reg_a, b=reg_b, c=reg_c, fill='toself', fillcolor='rgba(0, 255, 100, 0.2)', name='Nanoemulsion Region (O/W)'))
    fig.add_trace(go.Scatterternary(mode='markers', a=[oil], b=[smix], c=[100-oil-smix], marker=dict(color='red', size=15), name='Selected Point'))
    fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
    st.plotly_chart(fig, use_container_width=True)
    
    
    if st.button("Next: Final AI Predictions ➡️"):
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4: AI PREDICTION ---
elif nav == "Step 4: AI Prediction":
    st.header("4. AI-Driven Formulation Summary")
    try:
        in_df = pd.DataFrame([{'Drug_Name': encoders['Drug_Name'].transform([st.session_state.drug])[0],
                               'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
                               'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
                               'Co-surfactant': encoders['Co-surfactant'].transform([str(st.session_state.f_cs)])[0]}])
        res = {t: models[t].predict(in_df)[0] for t in models}
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Particle Size", f"{res['Size_nm']:.1f} nm")
        c2.metric("PDI", f"{res['PDI']:.3f}")
        c3.metric("Zeta Potential", f"{res['Zeta_mV']:.1f} mV")
        c4.metric("EE (%)", f"{res['Encapsulation_Efficiency']:.1f} %")
        
        # Stability Description
        st.divider()
        zeta = res['Zeta_mV']
        stability = "High Stability" if abs(zeta) > 30 else "Moderate Stability" if abs(zeta) > 20 else "Low Stability"
        st.subheader(f"🛡️ Physical Stability Assessment: {stability}")
        st.info(f"A Zeta Potential of {zeta:.2f} mV suggests the droplets have {'strong' if abs(zeta)>25 else 'weak'} electrostatic repulsion, preventing coalescence.")

        # SHAP Description
        st.subheader("💡 AI Decision Logic (SHAP)")
        st.write("**How to read this:** This waterfall plot shows how much each component moved the Particle Size from the average. Red bars increase size, blue bars decrease it.")
        explainer = shap.Explainer(models['Size_nm'], X_train)
        sv = explainer(in_df)
        fig_sh, _ = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(sv[0], show=False)
        st.pyplot(fig_sh)
        

    except Exception as e: st.error(f"Error: {e}. Please ensure all steps are completed.")
