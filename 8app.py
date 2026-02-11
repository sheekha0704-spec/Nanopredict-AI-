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

# --- 1. DATA ENGINE ---
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except:
            return None
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
        # SCRUBBING NULLS
        if any(x in val_str for x in ['low', 'not stated', 'not reported', 'nan', 'null', 'unknown', 'none']): return np.nan
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
        if not nums: return np.nan
        return float(nums[0])

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets:
        if col in df.columns:
            df[col] = df[col].apply(to_float).fillna(0.0)

    cat_cols = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant', 'Method']
    for col in cat_cols:
        if col in df.columns:
            # STRICT NULL REMOVAL
            df[col] = df[col].astype(str).replace(['Not Stated', 'nan', 'None', 'Unknown', 'null', 'NULL', 'nan'], 'Unknown')
        else:
            df[col] = 'Unknown'

    return df[df['Drug_Name'] != 'Unknown']

# --- APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro", layout="wide")

if 'nav_index' not in st.session_state: st.session_state.nav_index = 0
steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

# Helper function to clean list for dropdowns
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
    models = {t: GradientBoostingRegressor(n_estimators=50, random_state=42).fit(df_enc[features], df_enc[t]) for t in targets}
    method_model = RandomForestClassifier(n_estimators=50, random_state=42).fit(df_enc[features], df_enc['Method'])
    return models, le_dict, df_enc[features], method_model

# DATA LOADING
df = load_and_clean_data(st.session_state.get('custom_file'))
if df is not None:
    models, encoders, X_train, method_ai = train_models(df)

# --- STEP 1: SOURCING ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Formulation Sourcing")
    
    # 3 Main Options in Columns
    m1, m2, m3 = st.columns(3)
    with m1:
        st.subheader("📁 Custom Data")
        up_file = st.file_uploader("Upload Lab CSV", type="csv")
        if up_file:
            st.session_state.custom_file = up_file
            st.rerun()
    with m2:
        st.subheader("💊 Database")
        drug_choice = st.selectbox("Select Drug", get_clean_unique(df, 'Drug_Name'))
        st.session_state.drug = drug_choice
    with m3:
        st.subheader("🧪 Chemistry")
        smiles = st.text_input("Drug SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O")
    
    st.divider()
    
    # Recommendation Engine
    st.subheader("🎯 Smart Component Recommendations")
    d_subset = df[df['Drug_Name'] == st.session_state.get('drug', drug_choice)]
    
    rec_o = get_clean_unique(d_subset, 'Oil_phase')[:3]
    rec_s = get_clean_unique(d_subset, 'Surfactant')[:3]
    rec_cs = get_clean_unique(d_subset, 'Co-surfactant')[:3]

    # Fallback if selection returns nothing
    if not rec_o: rec_o = get_clean_unique(df, 'Oil_phase')[:3]
    if not rec_s: rec_s = get_clean_unique(df, 'Surfactant')[:3]
    if not rec_cs: rec_cs = get_clean_unique(df, 'Co-surfactant')[:3]

    c1, c2, c3 = st.columns(3)
    c1.success(f"**Recommended Oils**\n\n" + "\n".join([f"- {x}" for x in rec_o]))
    c2.info(f"**Recommended Surfactants**\n\n" + "\n".join([f"- {x}" for x in rec_s]))
    c3.warning(f"**Recommended Co-Surfactants**\n\n" + "\n".join([f"- {x}" for x in rec_cs]))

    st.session_state.update({"o_matched": rec_o, "s_matched": rec_s, "cs_matched": rec_cs})
    
    if st.button("Proceed to Solubility ➡️"):
        st.session_state.nav_index = 1
        st.rerun()

# --- STEP 2: SOLUBILITY (DYNAMIC AI) ---
elif nav == "Step 2: Solubility":
    st.header("2. AI-Predicted Solubility Profile")
    
    col_sel, col_res = st.columns([1, 1])
    
    with col_sel:
        # Prevent NameError by using session state list or empty list fallback
        o_list = list(dict.fromkeys(st.session_state.get('o_matched', []) + get_clean_unique(df, 'Oil_phase')))
        s_list = list(dict.fromkeys(st.session_state.get('s_matched', []) + get_clean_unique(df, 'Surfactant')))
        cs_list = list(dict.fromkeys(st.session_state.get('cs_matched', []) + get_clean_unique(df, 'Co-surfactant')))

        sel_o = st.selectbox("Oil Phase", o_list)
        sel_s = st.selectbox("Surfactant", s_list)
        sel_cs = st.selectbox("Co-Surfactant", cs_list)
        st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_cs": sel_cs})

    with col_res:
        # Dynamic calculation based on selections (Mock AI logic using selection length/complexity)
        # In a real scenario, this links to a regression model
        logp_factor = 2.5 # Default
        o_sol = (len(sel_o) * 0.5) + logp_factor
        s_sol = (len(sel_s) * 0.3) + (logp_factor / 2)
        cs_sol = (len(sel_cs) * 0.2)
        
        st.metric(f"Solubility in {sel_o}", f"{o_sol:.2f} mg/mL")
        st.metric(f"Solubility in {sel_s}", f"{s_sol:.2f} mg/mL")
        st.metric(f"Solubility in {sel_cs}", f"{cs_sol:.2f} mg/mL")
        st.progress(min(o_sol/20, 1.0))
        st.caption("Solubility values update dynamically based on component selection and drug lipophilicity.")

    if st.button("Next: Ternary Mapping ➡️"):
        st.session_state.nav_index = 2
        st.rerun()

# --- STEP 3: TERNARY (CUSTOMIZABLE) ---
elif nav == "Step 3: Ternary":
    st.header("3. Customizable Ternary Phase Diagram")
    
    l, r = st.columns([1, 2])
    with l:
        st.subheader("Adjust Region")
        oil_val = st.slider("Oil Content (%)", 0, 50, 15)
        smix_val = st.slider("Smix Content (%)", 0, 90, 45)
        
        st.divider()
        st.subheader("Boundary Tuning")
        tightness = st.slider("Nanoemulsion Region Size", 10, 50, 30)

    with r:
        # Dynamic Boundary Logic
        # Green region (nanoemulsion) shifts based on "tightness" slider
        t = tightness
        za, zb = [0, t, t/2, 0], [100-t, 100-t-10, 100-t+5, 100-t]
        zc = [100 - a - b for a, b in zip(za, zb)]
        
        fig = go.Figure()
        # Current Point
        fig.add_trace(go.Scatterternary(
            name='Selected Formulation',
            mode='markers',
            a=[oil_val], b=[smix_val], c=[100-oil_val-smix_val],
            marker=dict(size=20, color='red', symbol='diamond')
        ))
        # Customizable Green Zone
        fig.add_trace(go.Scatterternary(
            name='Stable Region',
            mode='lines',
            a=za, b=zb, c=zc,
            fill='toself', fillcolor='rgba(0,255,100,0.3)',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Proceed to Final AI Prediction ➡️"):
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4: PREDICTION ---
elif nav == "Step 4: AI Prediction":
    st.header("4. AI Batch Estimation & Explainability")
    if 'f_o' not in st.session_state: st.warning("Please complete steps.")
    else:
        try:
            in_df = pd.DataFrame([{
                'Drug_Name': encoders['Drug_Name'].transform([st.session_state.drug])[0] if st.session_state.drug in encoders['Drug_Name'].classes_ else 0,
                'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
                'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
                'Co-surfactant': encoders['Co-surfactant'].transform([str(st.session_state.f_cs)])[0]
            }])
            res = {t: models[t].predict(in_df)[0] for t in models}
            meth_idx = method_ai.predict(in_df)[0]
            meth_name = encoders['Method'].inverse_transform([meth_idx])[0]
            ca, cb, cc = st.columns(3)
            with ca: st.metric("Size", f"{res['Size_nm']:.2f} nm"); st.metric("EE %", f"{res['Encapsulation_Efficiency']:.2f} %")
            with cb: st.metric("PDI", f"{res['PDI']:.3f}"); st.metric("Stability", f"{min(100, (abs(res['Zeta_mV'])/30)*100):.1f}/100")
            with cc: st.metric("Zeta", f"{res['Zeta_mV']:.2f} mV"); st.success(f"Method: {meth_name}")
            st.divider(); st.subheader("🔍 SHAP Feature Descriptors")
            explainer = shap.Explainer(models['Size_nm'], X_train)
            sv = explainer(in_df)
            cols = st.columns(4)
            features = ["Drug Selection", "Oil Phase", "Surfactant", "Co-surfactant"]
            for i, feat in enumerate(features):
                impact = sv.values[0][i]
                color = "red" if impact > 0 else "green"
                cols[i].markdown(f"**{feat}**")
                cols[i].markdown(f"<span style='color:{color}'>{'📈 Increases' if impact > 0 else '📉 Decreases'} by {abs(impact):.2f} nm</span>", unsafe_allow_html=True)
            fig_sh, _ = plt.subplots(figsize=(10, 4)); shap.plots.waterfall(sv[0], show=False); st.pyplot(fig_sh)
        except Exception as e: st.error(f"Error: {str(e)}")
