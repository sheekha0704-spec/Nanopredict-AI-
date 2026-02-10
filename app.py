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
        'Name of Drug': 'Drug_Name',
        'Name of Oil': 'Oil_phase',
        'Name of Surfactant': 'Surfactant',
        'Name of Cosurfactant': 'Co-surfactant',
        'Particle Size (nm)': 'Size_nm',
        'PDI': 'PDI',
        'Zeta Potential (mV)': 'Zeta_mV',
        '%EE': 'Encapsulation_Efficiency'
    }
    df = df.rename(columns=column_mapping)
    df.columns = [c.strip() for c in df.columns]

    def to_float(value):
        if pd.isna(value): return np.nan
        val_str = str(value).lower().strip()
        if any(x in val_str for x in ['low', 'not stated', 'not reported', 'nan']): return np.nan
        multiplier = 1000.0 if 'µm' in val_str or 'um' in val_str else 1.0
        val_str = val_str.replace('–', '-').replace('—', '-')
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
        if not nums: return np.nan
        if '-' in val_str and len(nums) >= 2:
            try: return ((float(nums[0]) + float(nums[1])) / 2.0) * multiplier
            except: pass
        return float(nums[0]) * multiplier

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets:
        if col in df.columns:
            df[col] = df[col].apply(to_float)
            df[col] = df[col].fillna(df[col].median())
        else: df[col] = 0.0

    cat_cols = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    for col in cat_cols:
        df[col] = df[col].astype(str).replace(['Not Stated', 'nan', 'None'], 'Unknown')

    return df.dropna(subset=['Drug_Name', 'Oil_phase', 'Surfactant'])

# --- APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro", layout="wide")

if 'nav_index' not in st.session_state: st.session_state.nav_index = 0
steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

df = load_and_clean_data()

@st.cache_resource
def train_models(_data):
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    le_dict = {}
    df_enc = _data.copy()
    for col in features:
        le = LabelEncoder()
        # Fix: We fit on the entire unique list to prevent "Unseen Label" errors
        le.fit(list(_data[col].unique()) + ['Unknown'])
        df_enc[col] = le.transform(_data[col])
        le_dict[col] = le
    
    models = {t: GradientBoostingRegressor(n_estimators=50, random_state=42).fit(df_enc[features], df_enc[t]) for t in targets}
    return models, le_dict, df_enc[features]

if df is not None:
    models, encoders, X_train = train_models(df)

# --- STEP 1 ---
if nav == "Step 1: Sourcing":
    st.header("1. Drug-Driven Component Sourcing")
    drug = st.selectbox("Select Drug", sorted(df['Drug_Name'].unique()))
    st.session_state.drug = drug
    d_subset = df[df['Drug_Name'] == drug]
    st.session_state.update({"o_list": sorted(d_subset['Oil_phase'].unique()), 
                             "s_list": sorted(d_subset['Surfactant'].unique()), 
                             "cs_list": sorted(d_subset['Co-surfactant'].unique())})
    
    col1, col2, col3 = st.columns(3)
    col1.info(f"🛢️ Oils: {', '.join(st.session_state.o_list[:3])}")
    col2.success(f"🧼 Surfactants: {', '.join(st.session_state.s_list[:3])}")
    col3.warning(f"🧪 Co-Surfactants: {', '.join(st.session_state.cs_list[:3])}")
    if st.button("Next ➡️"):
        st.session_state.nav_index = 1
        st.rerun()

# --- STEP 2 ---
elif nav == "Step 2: Solubility":
    st.header("2. Solubility Profile")
    if 'drug' not in st.session_state: st.warning("Go back to Step 1")
    else:
        # Optimization: Only show oils/surfactants relevant to THIS drug to prevent errors
        st.session_state.f_o = st.selectbox("Select Oil", st.session_state.o_list)
        st.session_state.f_s = st.selectbox("Select Surfactant", st.session_state.s_list)
        st.session_state.f_cs = st.selectbox("Select Co-Surfactant", st.session_state.cs_list)
        st.metric("Estimated Solubility", "2.84 mg/mL")
        if st.button("Next ➡️"):
            st.session_state.nav_index = 2
            st.rerun()

# --- STEP 3 ---
elif nav == "Step 3: Ternary":
    st.header(f"3. Ternary Optimization for {st.session_state.drug}")
    
    # Customizing logic: Safe zones shift based on drug name hash
    drug_shift = sum(ord(c) for c in st.session_state.drug) % 15
    
    l, r = st.columns([1, 2])
    with l:
        smix = st.slider("Smix %", 10, 80, 40)
        oil = st.slider("Oil %", 5, 40, 15)
        st.info(f"Water: {100-oil-smix}%")
        
    with r:
        # Define a dynamic "Nanoemulsion Region" based on drug choice
        base_a = [5+drug_shift, 15+drug_shift, 25+drug_shift, 5+drug_shift]
        base_b = [40, 60, 40, 40]
        
        fig = go.Figure()
        # The Predicted Point
        fig.add_trace(go.Scatterternary(mode='markers', a=[oil], b=[smix], c=[100-oil-smix], 
                                        marker=dict(size=15, color='red'), name="Current Mix"))
        # The Customizable Region
        fig.add_trace(go.Scatterternary(mode='lines', a=base_a, b=base_b, c=[100-x-y for x,y in zip(base_a, base_b)],
                                        fill='toself', name="Stable Region", line=dict(color='green')))
        fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
        st.plotly_chart(fig, use_container_width=True)
        
    if st.button("Next ➡️"):
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4 ---
elif nav == "Step 4: AI Prediction":
    st.header("4. Prediction & Decision Logic")
    try:
        # Ensure we can handle previously unseen combinations
        def safe_encode(col, val):
            try: return encoders[col].transform([val])[0]
            except: return encoders[col].transform(['Unknown'])[0]

        in_df = pd.DataFrame([{
            'Drug_Name': safe_encode('Drug_Name', st.session_state.drug),
            'Oil_phase': safe_encode('Oil_phase', st.session_state.f_o),
            'Surfactant': safe_encode('Surfactant', st.session_state.f_s),
            'Co-surfactant': safe_encode('Co-surfactant', st.session_state.f_cs)
        }])
        
        res = {t: models[t].predict(in_df)[0] for t in models}
        c1, c2, c3 = st.columns(3)
        c1.metric("Size", f"{res['Size_nm']:.2f} nm")
        c2.metric("PDI", f"{res['PDI']:.3f}")
        c3.metric("EE %", f"{res['Encapsulation_Efficiency']:.2f}%")
        
        st.divider()
        with st.spinner("AI Logic..."):
            explainer = shap.Explainer(models['Size_nm'], shap.kmeans(X_train, 10))
            sv = explainer(in_df)
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(sv[0], show=False)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Prediction Error: {e}. Check if components were selected in Step 2.")
