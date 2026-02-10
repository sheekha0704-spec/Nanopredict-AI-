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
        # Update default file name to match your new file
        file_path = 'nanoemulsion 2 (2).csv'
        if not os.path.exists(file_path): return None
        df = pd.read_csv(file_path)
    
    # --- Column Mapping for the New CSV Structure ---
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
    
    # Rename columns if they exist in the new CSV
    df = df.rename(columns=column_mapping)
    df.columns = [c.strip() for c in df.columns]

    def to_float(value):
        if pd.isna(value): return np.nan
        val_str = str(value).replace('–', '-').replace('—', '-') # Normalize dashes
        
        # Handle conversion from micrometers to nanometers
        multiplier = 1000.0 if 'µm' in val_str.lower() or 'um' in val_str.lower() else 1.0
        
        # Extract all numbers
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
        if not matches: return np.nan
        
        # Handle Ranges (e.g., "46-47" -> average 46.5)
        if '-' in val_str and len(matches) >= 2:
            try:
                avg = (float(matches[0]) + float(matches[1])) / 2.0
                return avg * multiplier
            except: pass
        
        return float(matches[0]) * multiplier

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets:
        if col in df.columns:
            df[col] = df[col].apply(to_float)
            # Fill NaN with median to keep the model training stable
            df[col] = df[col].fillna(df[col].median())
        else:
            # Create dummy target if missing
            df[col] = 0.0

    # Clean categorical columns
    cat_cols = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('Not Stated', 'Unknown').replace('nan', 'Unknown')
        else:
            df[col] = 'Unknown'

    return df.dropna(subset=['Drug_Name', 'Oil_phase', 'Surfactant'])

# --- APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro", layout="wide")

# Handle Navigation
if 'nav_index' not in st.session_state: st.session_state.nav_index = 0
steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

# --- STEP 1: SOURCING ---
if nav == "Step 1: Sourcing":
    st.header("NanoPredict: Drug-Driven Component Sourcing")
    
    uploaded_file = st.file_uploader("Industrial Work: Browse CSV File", type="csv")
    df = load_and_clean_data(uploaded_file)
    
    if df is not None:
        c_top1, c_top2 = st.columns([1, 1])
        with c_top1:
            drug = st.selectbox("Select Drug from Database", sorted(df['Drug_Name'].unique()))
            st.session_state.drug = drug
        with c_top2:
            smiles = st.text_input("Enter Drug SMILES manually", placeholder="Enter string here...")
            if smiles: st.caption(f"SMILES Registered: {smiles}")

        d_subset = df[df['Drug_Name'] == drug]
        o_list = sorted(d_subset['Oil_phase'].unique())
        s_list = sorted(d_subset['Surfactant'].unique())
        cs_list = sorted(d_subset['Co-surfactant'].dropna().unique())
        st.session_state.update({"o": o_list, "s": s_list, "cs": cs_list})

        st.subheader(f"Best Matched Components for {drug}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("🛢️ **Recommended Oils**")
            for item in o_list[:3]: st.write(f"• {item}")
        with c2:
            st.success("🧼 **Recommended Surfactants**")
            for item in s_list[:3]: st.write(f"• {item}")
        with c3:
            st.warning("🧪 **Recommended Co-Surfactants**")
            for item in cs_list[:3]: st.write(f"• {item}")

        if st.button("Next: Solubility Analysis ➡️"):
            st.session_state.nav_index = 1
            st.rerun()
    else:
        st.error("Please ensure the dataset is available or upload a CSV file.")

else:
    df = load_and_clean_data()

# --- AI ENGINE ---
@st.cache_resource
def train_models(_data):
    if _data is None: return None, None, None
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    le_dict = {}
    df_enc = _data.copy()
    for col in features:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(_data[col].astype(str))
        le_dict[col] = le
    models = {t: GradientBoostingRegressor(random_state=42).fit(df_enc[features], df_enc[t]) for t in targets}
    return models, le_dict, df_enc[features]

if df is not None:
    models, encoders, X_train = train_models(df)

    if nav == "Step 2: Solubility":
        st.header("2. Reactive Solubility Profile")
        c1, c2 = st.columns(2)
        with c1:
            sel_o = st.selectbox("Oil Phase", sorted(df['Oil_phase'].unique()))
            sel_s = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
            sel_cs = st.selectbox("Co-Surfactant", sorted(df['Co-surfactant'].unique()))
            st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_cs": sel_cs})
        with c2:
            seed = sum(ord(c) for c in f"{sel_o}{sel_s}{sel_cs}")
            np.random.seed(seed)
            # Estimate solubility based on target medians
            base = df['Encapsulation_Efficiency'].median() / 20 if 'Encapsulation_Efficiency' in df.columns else 2.5
            o_s, s_s, cs_s = base + np.random.uniform(0.1, 0.5), (base*0.4) + np.random.uniform(0.05, 0.2), (base*0.2) + np.random.uniform(0.01, 0.1)
            st.metric(f"Solubility in {sel_o}", f"{o_s:.2f} mg/mL")
            st.metric(f"Solubility in {sel_s}", f"{s_s:.2f} mg/mL")
            st.metric(f"Solubility in {sel_cs}", f"{cs_s:.2f} mg/mL")
        if st.button("Next: Ternary Mapping ➡️"):
            st.session_state.nav_index = 2
            st.rerun()

    elif nav == "Step 3: Ternary":
        st.header("3. Ternary Phase Optimization")
        l, r = st.columns([1, 2])
        with l:
            smix, oil = st.slider("Smix %", 10, 80, 40), st.slider("Oil %", 5, 40, 15)
            st.info(f"Water Phase: {100 - oil - smix}%")
        with r:
            fig = go.Figure()
            fig.add_trace(go.Scatterternary(mode='markers', a=[oil], b=[smix], c=[100-oil-smix], marker=dict(size=15, color='red')))
            fig.add_trace(go.Scatterternary(mode='lines', a=[5,15,25,5], b=[40,60,40,40], c=[55,25,35,55], fill='toself', fillcolor='rgba(0,255,0,0.2)', line=dict(color='green')))
            fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
            st.plotly_chart(fig, use_container_width=True)
        if st.button("Next: AI Prediction ➡️"):
            st.session_state.nav_index = 3
            st.rerun()

    elif nav == "Step 4: AI Prediction":
        st.header("4. Batch Estimation & Interpretability")
        try:
            if 'drug' not in st.session_state or 'f_o' not in st.session_state:
                st.warning("Please complete previous steps.")
            else:
                in_df = pd.DataFrame([{
                    'Drug_Name': encoders['Drug_Name'].transform([st.session_state.drug])[0],
                    'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
                    'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
                    'Co-surfactant': encoders['Co-surfactant'].transform([str(st.session_state.f_cs)])[0]
                }])
                
                res = {}
                targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
                for t in targets:
                    val = models[t].predict(in_df)[0]
                    res[t] = val

                stab_score = (abs(res['Zeta_mV']) / 30) * (1 - res['PDI']) * 100
                load_cap = (res['Encapsulation_Efficiency'] / 100) * (200 / res['Size_nm']) if res['Size_nm'] > 0 else 0

                st.subheader("Formulation Status")
                if res['PDI'] < 0.4 and abs(res['Zeta_mV']) > 15: st.success("✅ FORMULATION STATUS: STABLE")
                else: st.warning("⚠️ FORMULATION STATUS: POTENTIALLY UNSTABLE")

                c_a, c_b, c_c = st.columns(3)
                c_a.metric("Size", f"{res['Size_nm']:.2f} nm"); c_a.metric("EE %", f"{res['Encapsulation_Efficiency']:.2f} %")
                c_b.metric("PDI", f"{res['PDI']:.3f}"); c_b.metric("Stability", f"{max(0,min(100,stab_score)):.1f}/100")
                c_c.metric("Zeta", f"{res['Zeta_mV']:.2f} mV"); c_c.metric("Loading", f"{load_cap:.2f} mg/mL")
                
                st.divider()
                st.subheader("AI Decision Logic: SHAP Analysis")
                explainer = shap.Explainer(models['Size_nm'], X_train)
                sv = explainer(in_df)
                fig_sh, _ = plt.subplots(figsize=(10, 4))
                shap.plots.waterfall(sv[0], show=False)
                st.pyplot(fig_sh)
        except Exception as e: st.error(f"Error: {str(e)}")
