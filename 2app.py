import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import os
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v2.0", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .metric-card { background: #f8f9fa; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid #28a745; text-align: center; }
    .m-label { font-size: 14px; color: #555; font-weight: bold; margin-bottom: 5px; }
    .m-value { font-size: 24px; font-weight: 800; color: #1a202c; }
    .status-box { padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_resource
def load_and_prep():
    csv_path = 'nanoemulsion 2 (2).csv'
    if not os.path.exists(csv_path):
        return None, None, None, None, None
    
    df = pd.read_csv(csv_path)
    
    # Mapping and Cleaning
    def clean_numeric(x):
        if pd.isna(x) or str(x).lower() == 'not stated' or str(x).lower() == 'low':
            return 0.0
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        if nums:
            # If range, take average
            vals = [float(n) for n in nums]
            return sum(vals) / len(vals)
        return 0.0

    target_map = {
        'Size_nm': 'Particle Size (nm)',
        'PDI': 'PDI',
        'Zeta_mV': 'Zeta Potential (mV)',
        'EE': '%EE'
    }
    
    for key, col in target_map.items():
        df[f'{key}_clean'] = df[col].apply(clean_numeric)

    cat_cols = ['Name of Drug', 'Name of Oil', 'Name of Surfactant', 'Name of Cosurfactant']
    for col in cat_cols:
        df[col] = df[col].fillna("Not Stated").astype(str).str.strip()

    # Feature Engineering
    le_dict = {}
    df_train = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le

    features = [f'{c}_enc' for c in cat_cols]
    X = df_train[features]
    
    # Models
    models = {key: GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X, df_train[f'{key}_clean']) 
              for key in target_map.keys()}
    
    df_train['is_stable'] = df_train['Stability'].str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(random_state=42).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict, X

df, models, stab_model, le_dict, X_train = load_and_prep()

def get_enc(le, val):
    try:
        return le.transform([val])[0]
    except:
        return 0

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("NanoPredict AI")
    step = st.radio("Navigation", ["Step 1: Drug Selection", "Step 2: Component Solubility", "Step 3: Ternary Phase Diagram", "Step 4: Final Predictions"])

# --- STEP 1: DRUG SELECTION ---
if step == "Step 1: Drug Selection":
    st.header("Step 1: Drug Identification & Compatibility")
    method = st.radio("Selection Method", ["Select from Dataset", "Manual Entry", "SMILES Prediction"])
    
    drug_name = "Not Stated"
    logp, mw = 3.0, 300.0

    if method == "Select from Dataset":
        drug_name = st.selectbox("Choose Drug", sorted(df['Name of Drug'].unique()))
    elif method == "Manual Entry":
        drug_name = st.text_input("Enter Drug Name", "Ibuprofen")
    else:
        smiles = st.text_input("Enter SMILES", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            logp, mw = round(Descriptors.MolLogP(mol), 2), round(Descriptors.MolWt(mol), 2)
            st.image(Draw.MolToImage(mol, size=(300, 300)))
            st.info(f"Calculated Properties: LogP: {logp}, Molecular Weight: {mw}")

    st.session_state['drug_name'] = drug_name
    st.session_state['logp'] = logp

    # AI Recommendation
    st.subheader("💡 Suggested Components")
    if drug_name in df['Name of Drug'].values:
        best_rec = df[df['Name of Drug'] == drug_name].sort_values('Size_nm_clean').iloc[0]
        st.success(f"Recommended Oil: **{best_rec['Name of Oil']}**")
        st.success(f"Recommended Surfactant: **{best_rec['Name of Surfactant']}**")
    else:
        # Generic suggestion based on LogP
        st.warning("New Drug: Suggested Oil: MCT Oil or Oleic Acid (High LogP compatible)")

# --- STEP 2: COMPONENT SOLUBILITY ---
elif step == "Step 2: Component Selection":
    st.header("Step 2: Excipient Selection & Solubility")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        oil = st.selectbox("Select Oil Phase", sorted(le_dict['Name of Oil'].classes_))
    with col2:
        surf = st.selectbox("Select Surfactant", sorted(le_dict['Name of Surfactant'].classes_))
    with col3:
        cosurf = st.selectbox("Select Co-Surfactant", sorted(le_dict['Name of Cosurfactant'].classes_))
    
    st.session_state.update({'oil': oil, 'surf': surf, 'cosurf': cosurf})

    # Solubility logic
    logp = st.session_state.get('logp', 3.0)
    sol_water = 10**(0.5 - 0.6 * logp) * 100
    sol_oil = sol_water * (10**logp)
    
    st.write("---")
    c1, c2 = st.columns(2)
    c1.metric("Predicted Solubility in Oil", f"{sol_oil:.2f} mg/mL")
    c2.metric("Predicted Solubility in Water", f"{sol_water:.4f} mg/mL")

# --- STEP 3: TERNARY PHASE DIAGRAM ---
elif step == "Step 3: Ternary Phase Diagram":
    st.header("Step 3: Ternary Phase & Ratios")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Formulation Ratios")
        oil_p = st.slider("Oil %", 1, 50, 15)
        smix_p = st.slider("S-mix % (Surfactant + Cosurf)", 10, 80, 40)
        water_p = 100 - oil_p - smix_p
        
        km = st.selectbox("Smix Ratio (S:CoS)", ["1:1", "2:1", "3:1", "4:1"])
        st.session_state.update({'oil_p': oil_p, 'smix_p': smix_p, 'water_p': water_p, 'km': km})

    with col2:
        fig = go.Figure(go.Scatterternary({
            'mode': 'markers',
            'a': [oil_p], 'b': [smix_p], 'c': [water_p],
            'marker': {'size': 25, 'color': '#28a745', 'symbol': 'diamond'}
        }))
        fig.update_layout(ternary={'aaxis':{'title':'Oil'}, 'baxis':{'title':'Smix'}, 'caxis':{'title':'Water'}})
        st.plotly_chart(fig, use_container_width=True)

# --- STEP 4: FINAL PREDICTIONS ---
elif step == "Step 4: Final Predictions":
    st.header("Step 4: AI Output & SHAP Analysis")
    
    input_data = pd.DataFrame([{
        'Name of Drug_enc': get_enc(le_dict['Name of Drug'], st.session_state.get('drug_name', 'Not Stated')),
        'Name of Oil_enc': get_enc(le_dict['Name of Oil'], st.session_state.get('oil', 'Not Stated')),
        'Name of Surfactant_enc': get_enc(le_dict['Name of Surfactant'], st.session_state.get('surf', 'Not Stated')),
        'Name of Cosurfactant_enc': get_enc(le_dict['Name of Cosurfactant'], st.session_state.get('cosurf', 'Not Stated'))
    }])

    # 1. Stability
    stable = stab_model.predict(input_data)[0]
    color = "#d4edda" if stable else "#f8d7da"
    label = "STABLE FORMULATION" if stable else "POTENTIALLY UNSTABLE"
    st.markdown(f'<div class="status-box" style="background-color: {color};">{label}</div>', unsafe_allow_html=True)

    # 2. Predicted Metrics
    cols = st.columns(4)
    metrics = [('Size_nm', 'Size (nm)'), ('PDI', 'PDI'), ('Zeta_mV', 'Zeta (mV)'), ('EE', 'EE (%)')]
    
    for i, (key, label) in enumerate(metrics):
        val = models[key].predict(input_data)[0]
        with cols[i]:
            st.markdown(f"<div class='metric-card'><div class='m-label'>{label}</div><div class='m-value'>{val:.3f}</div></div>", unsafe_allow_html=True)

    # 3. SHAP Explanation
    st.write("---")
    st.subheader("💡 AI Decision Logic (SHAP)")
    explainer = shap.Explainer(models['Size_nm'], X_train)
    shap_values = explainer(input_data)
    
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())
