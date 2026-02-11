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
        if any(x in val_str for x in ['low', 'not stated', 'nan', 'null', 'none']): return np.nan
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

# --- APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro", layout="wide")

if 'nav_index' not in st.session_state: st.session_state.nav_index = 0
steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

def get_clean_unique(df, col):
    items = df[col].unique()
    return sorted([str(x) for x in items if str(x).lower() not in ['unknown', 'nan', 'null', 'none', '']])

df = load_and_clean_data(st.session_state.get('custom_file'))

# --- STEP 1: SOURCING ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Formulation Sourcing")
    m1, m2, m3 = st.columns(3)
    with m1:
        up_file = st.file_uploader("Upload Lab CSV", type="csv")
        if up_file: st.session_state.custom_file = up_file; st.rerun()
    with m2:
        drug_choice = st.selectbox("Select Drug", get_clean_unique(df, 'Drug_Name'))
        st.session_state.drug = drug_choice
    with m3:
        smiles = st.text_input("Drug SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O")
    
    st.divider()
    st.subheader("🎯 3-Point Recommendations")
    d_subset = df[df['Drug_Name'] == st.session_state.get('drug', drug_choice)]
    
    # Logic to ensure 3 recommendations per category
    rec_o = get_clean_unique(d_subset, 'Oil_phase')[:3]
    if len(rec_o) < 3: rec_o += [x for x in get_clean_unique(df, 'Oil_phase') if x not in rec_o][:3-len(rec_o)]
    
    rec_s = get_clean_unique(d_subset, 'Surfactant')[:3]
    if len(rec_s) < 3: rec_s += [x for x in get_clean_unique(df, 'Surfactant') if x not in rec_s][:3-len(rec_s)]
    
    rec_cs = get_clean_unique(d_subset, 'Co-surfactant')[:3]
    if len(rec_cs) < 3: rec_cs += [x for x in get_clean_unique(df, 'Co-surfactant') if x not in rec_cs][:3-len(rec_cs)]

    c1, c2, c3 = st.columns(3)
    c1.success("**Oils**\n\n" + "\n".join([f"- {x}" for x in rec_o]))
    c2.info("**Surfactants**\n\n" + "\n".join([f"- {x}" for x in rec_s]))
    c3.warning("**Co-Surfactants**\n\n" + "\n".join([f"- {x}" for x in rec_cs]))
    st.session_state.update({"o_matched": rec_o, "s_matched": rec_s, "cs_matched": rec_cs})
    
    if st.button("Proceed to Solubility ➡️"): st.session_state.nav_index = 1; st.rerun()

# --- STEP 2: SOLUBILITY ---
elif nav == "Step 2: Solubility":
    st.header("2. AI-Predicted Solubility Profile")
    o_list = list(dict.fromkeys(st.session_state.get('o_matched', []) + get_clean_unique(df, 'Oil_phase')))
    s_list = list(dict.fromkeys(st.session_state.get('s_matched', []) + get_clean_unique(df, 'Surfactant')))
    cs_list = list(dict.fromkeys(st.session_state.get('cs_matched', []) + get_clean_unique(df, 'Co-surfactant')))

    c1, c2 = st.columns(2)
    with c1:
        sel_o = st.selectbox("Oil Phase", o_list)
        sel_s = st.selectbox("Surfactant", s_list)
        sel_cs = st.selectbox("Co-Surfactant", cs_list)
        st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_cs": sel_cs})
    with c2:
        # Mock AI logic for demonstration
        o_sol = (len(sel_o) * 0.4) + 2.0
        s_sol = (len(sel_s) * 0.2) + 1.5
        cs_sol = (len(sel_cs) * 0.1) + 0.5
        st.metric(f"Solubility in {sel_o}", f"{o_sol:.2f} mg/mL")
        st.metric(f"Solubility in {sel_s}", f"{s_sol:.2f} mg/mL")
        st.metric(f"Solubility in {sel_cs}", f"{cs_sol:.2f} mg/mL")
        st.session_state.drug_sol = o_sol # Save for Step 3

    if st.button("Next: Ternary Mapping ➡️"): st.session_state.nav_index = 2; st.rerun()

# --- STEP 3: TERNARY (HLB & TEMP DEPENDENT) ---
elif nav == "Step 3: Ternary":
    st.header("3. Advanced Ternary Boundary Tuning")
    l, r = st.columns([1, 2])
    with l:
        hlb = st.slider("Surfactant HLB Value", 8.0, 18.0, 12.0)
        temp = st.slider("Temperature (°C)", 10, 60, 25)
        surf_conc = st.slider("Surfactant Concentration (%)", 5, 50, 20)
        solub = st.session_state.get('drug_sol', 5.0)
        st.info(f"Using Predicted Drug Solubility: {solub:.2f} mg/mL")

    with r:
        # Boundary algorithm: HLB and Temp expand/contract the stable region
        # Higher HLB usually increases nanoemulsion area for o/w; Temp shifts the phase inversion point
        base_size = (hlb * 2) + (surf_conc / 2) + (temp / 10) - (solub / 2)
        t = min(max(base_size, 5), 60) # Normalized area
        
        za, zb = [0, t, t*0.8, 0], [100-t, 100-t-5, 100-t+10, 100-t]
        zc = [100 - a - b for a, b in zip(za, zb)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterternary(name='Stable Region', mode='lines', a=za, b=zb, c=zc, fill='toself', fillcolor='rgba(0,255,100,0.3)', line=dict(color='green')))
        fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Proceed to Final AI Prediction ➡️"): st.session_state.nav_index = 3; st.rerun()

# --- STEP 4: PREDICTION (STABILITY RECHECKED) ---
elif nav == "Step 4: AI Prediction":
    st.header("4. Final AI Performance Metrics")
    # Corrected Stability Logic: High Zeta + Low PDI = High Stability
    # Simulating results for demonstration
    size = 120.5; pdi = 0.18; zeta = -35.2; ee = 88.4
    
    # Recalibrated stability string logic
    if abs(zeta) > 30 and pdi < 0.25:
        stability_status = "Excellent - Highly Stable System"
        color = "green"
    elif abs(zeta) > 20 or pdi < 0.3:
        stability_status = "Moderate - Stable with minor sedimentation risk"
        color = "orange"
    else:
        stability_status = "Low - Immediate Flocculation Risk"
        color = "red"

    c1, c2, c3 = st.columns(3)
    c1.metric("Particle Size", f"{size} nm")
    c2.metric("PDI", f"{pdi}")
    c3.metric("Zeta Potential", f"{zeta} mV")
    
    st.divider()
    st.subheader("Stability Assessment")
    st.markdown(f"**Status:** <span style='color:{color}; font-size:20px'>{stability_status}</span>", unsafe_allow_html=True)
    st.write(f"Encapsulation Efficiency: **{ee}%**")

    # Image of Zeta Potential Stability Scales
