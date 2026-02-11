import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import os
import re

# --- ROBUST RDKIT IMPORTS (Fix for Line 10) ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw
    from PIL import Image
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- HELPER FUNCTIONS ---
def clean_numeric_value(x):
    if pd.isna(x) or str(x).lower() in ['not stated', 'low', 'unknown']:
        return 0.0
    # Cleans LaTeX symbols ($), %, ±, and ranges (46–47)
    clean_s = str(x).replace('$', '').replace('%', '').replace('±', '+/-').replace('\\pm', '+/-')
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", clean_s)
    if nums:
        vals = [float(n) for n in nums]
        return sum(vals) / len(vals)
    return 0.0

def get_enc(le, val):
    try:
        return le.transform([str(val)])[0]
    except:
        return 0

# --- DATA ENGINE ---
@st.cache_resource
def load_data():
    file_path = 'nanoemulsion 2 (2).csv'
    if not os.path.exists(file_path):
        return None, None, None, None, None
    
    df = pd.read_csv(file_path)
    # Matching your specific CSV column names
    target_cols = {'Size_nm': 'Particle Size (nm)', 'PDI': 'PDI', 'Zeta_mV': 'Zeta Potential (mV)', 'EE': '%EE'}
    
    for key, col in target_cols.items():
        df[f'{key}_clean'] = df[col].apply(clean_numeric_value)

    cat_cols = ['Name of Drug', 'Name of Oil', 'Name of Surfactant', 'Name of Cosurfactant']
    le_dict = {}
    df_train = df.copy()
    
    for col in cat_cols:
        df_train[col] = df_train[col].fillna("Not Stated").astype(str)
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le

    features = [f'{col}_enc' for col in cat_cols]
    X = df_train[features]
    
    models = {key: GradientBoostingRegressor(random_state=42).fit(X, df_train[f'{key}_clean']) for key in target_cols.keys()}
    df_train['is_stable'] = df_train['Stability'].str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(random_state=42).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict, X

df_raw, models, stab_model, le_dict, X_train = load_data()

# --- APP LAYOUT ---
st.title("🔬 NanoPredict AI v3.3")
step = st.sidebar.radio("Navigation", ["Step 1: Drug Selection", "Step 2: Component Solubility", "Step 3: Ternary Phase", "Step 4: AI Analysis"])

# --- STEP 1: DRUG SELECTION ---
if step == "Step 1: Drug Selection":
    st.header("1️⃣ Drug Selection & Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if not RDKIT_AVAILABLE:
            st.error("RDKit is not installed correctly. Please run: pip install rdkit")
        
        method = st.radio("Selection Method", ["Database", "SMILES"])
        if method == "Database":
            drug = st.selectbox("Select Drug", sorted(df_raw['Name of Drug'].unique()))
            st.session_state['drug_name'] = drug
        elif method == "SMILES" and RDKIT_AVAILABLE:
            smiles = st.text_input("Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Convert RDKit Draw result to a PIL image for Streamlit
                img = Draw.MolToImage(mol, size=(300, 300))
                st.image(img, caption="Drug Structure")
                st.session_state['logp'] = round(Descriptors.MolLogP(mol), 2)
                st.session_state['drug_name'] = "Custom"
            else:
                st.warning("Invalid SMILES string.")

    with col2:
        st.subheader("AI Recommendation")
        if st.session_state.get('drug_name') in df_raw['Name of Drug'].values:
            history = df_raw[df_raw['Name of Drug'] == st.session_state['drug_name']].sort_values('Size_nm_clean')
            st.success(f"Recommended Oil: {history.iloc[0]['Name of Oil']}")
            st.success(f"Recommended Surfactant: {history.iloc[0]['Name of Surfactant']}")

# --- STEP 2: SOLUBILITY ---
elif step == "Step 2: Component Solubility":
    st.header("2️⃣ Selection & Solubility")
    c1, c2, c3 = st.columns(3)
    st.session_state.oil = c1.selectbox("Oil", sorted(le_dict['Name of Oil'].classes_))
    st.session_state.surf = c2.selectbox("Surfactant", sorted(le_dict['Name of Surfactant'].classes_))
    st.session_state.cosurf = c3.selectbox("Co-Surfactant", sorted(le_dict['Name of Cosurfactant'].classes_))
    
    logp = st.session_state.get('logp', 3.0)
    st.metric("Solubility Score", round(logp * 15, 1))

# --- STEP 3: TERNARY PHASE ---
elif step == "Step 3: Ternary Phase":
    st.header("3️⃣ Ternary Phase Map")
    o = st.slider("Oil %", 1, 50, 15)
    s = st.slider("Smix %", 10, 80, 40)
    w = 100 - o - s
    st.session_state.update({'o':o, 's':s, 'w':w})
    
    fig = go.Figure(go.Scatterternary({
        'mode': 'markers', 'a': [o], 'b': [s], 'c': [w], 
        'marker': {'size': 20, 'color': 'darkblue'}
    }))
    st.plotly_chart(fig, use_container_width=True)
    

# --- STEP 4: OUTPUT & SHAP ---
elif step == "Step 4: AI Analysis":
    st.header("4️⃣ Final Predictions & Analysis")
    
    input_df = pd.DataFrame([{
        'Name of Drug_enc': get_enc(le_dict['Name of Drug'], st.session_state.get('drug_name', 'Not Stated')),
        'Name of Oil_enc': get_enc(le_dict['Name of Oil'], st.session_state.get('oil', 'Not Stated')),
        'Name of Surfactant_enc': get_enc(le_dict['Name of Surfactant'], st.session_state.get('surf', 'Not Stated')),
        'Name of Cosurfactant_enc': get_enc(le_dict['Name of Cosurfactant'], st.session_state.get('cosurf', 'Not Stated'))
    }])

    stable = stab_model.predict(input_df)[0]
    st.info("STABILITY PREDICTION: " + ("STABLE" if stable else "UNSTABLE"))

    m_cols = st.columns(4)
    targets = [('Size_nm', 'Size'), ('PDI', 'PDI'), ('Zeta_mV', 'Zeta'), ('EE', '%EE')]
    for i, (k, lbl) in enumerate(targets):
        val = models[k].predict(input_df)[0]
        m_cols[i].metric(lbl, round(val, 3))

    st.write("---")
    st.subheader("Model Decision Logic (SHAP Analysis)")
    explainer = shap.Explainer(models['Size_nm'], X_train)
    shap_v = explainer(input_df)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_v[0], show=False)
    st.pyplot(fig)
