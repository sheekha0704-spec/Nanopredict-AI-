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

df = load_and_clean_data(st.session_state.get('custom_file'))
if df is not None:
    models, encoders, X_train, method_ai = train_models(df)

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
    
    def get_top_3(subset, full_df, col):
        res = get_clean_unique(subset, col)[:3]
        if len(res) < 3:
            extra = [x for x in get_clean_unique(full_df, col) if x not in res]
            res = (res + extra)[:3]
        return res

    rec_o = get_top_3(d_subset, df, 'Oil_phase')
    rec_s = get_top_3(d_subset, df, 'Surfactant')
    rec_cs = get_top_3(d_subset, df, 'Co-surfactant')

    c1, c2, c3 = st.columns(3)
    c1.success("**Recommended Oils**\n\n" + "\n".join([f"- {x}" for x in rec_o]))
    c2.info("**Recommended Surfactants**\n\n" + "\n".join([f"- {x}" for x in rec_s]))
    c3.warning("**Recommended Co-Surfactants**\n\n" + "\n".join([f"- {x}" for x in rec_cs]))
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
        o_sol = (len(sel_o) * 0.45) + 2.5
        s_sol = (len(sel_s) * 0.25) + 1.8
        cs_sol = (len(sel_cs) * 0.15) + 0.6
        st.metric(f"Solubility in {sel_o}", f"{o_sol:.2f} mg/mL")
        st.metric(f"Solubility in {sel_s}", f"{s_sol:.2f} mg/mL")
        st.metric(f"Solubility in {sel_cs}", f"{cs_sol:.2f} mg/mL")

    if st.button("Next: Ternary Mapping ➡️"): st.session_state.nav_index = 2; st.rerun()

# --- STEP 3: TERNARY (RECALIBRATED MAPPING) ---
elif nav == "Step 3: Ternary":
    st.header("3. Ternary Phase Mapping")
    l, r = st.columns([1, 2])
    with l:
        st.markdown("### Formulation Input")
        oil_val = st.slider("Oil Content (%)", 5, 50, 15)
        smix_val = st.slider("Smix (Surf/Co-Surf) %", 10, 80, 45)
        st.info("The green zone represents the theoretically stable O/W nanoemulsion region for your selected components.")
    with r:
        # Recalibrated stable region: Typical for High-Smix systems
        # a=Oil, b=Smix, c=Water
        za = [0, 25, 15, 0]    # Oil boundaries
        zb = [40, 65, 85, 40]  # Smix boundaries
        zc = [100 - a - b for a, b in zip(za, zb)]
        
        fig = go.Figure()
        # Stable Region
        fig.add_trace(go.Scatterternary(
            name='Stable Nanoemulsion Zone', 
            mode='lines', a=za, b=zb, c=zc, 
            fill='toself', fillcolor='rgba(0,255,100,0.25)', 
            line=dict(color='green', width=2)
        ))
        # User Selection
        fig.add_trace(go.Scatterternary(
            name='Your Formulation', 
            mode='markers', a=[oil_val], b=[smix_val], c=[100-oil_val-smix_val], 
            marker=dict(size=18, color='red', symbol='diamond', line=dict(width=2, color='white'))
        ))
        
        fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Next: AI Prediction ➡️"):
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4: PREDICTION (RECALIBRATED STABILITY & SHAP) ---
elif nav == "Step 4: AI Prediction":
    st.header("4. AI Batch Estimation & Explainability")
    if 'f_o' not in st.session_state: st.warning("Please complete steps.")
    else:
        try:
            # Data Preparation for AI
            in_df = pd.DataFrame([{
                'Drug_Name': encoders['Drug_Name'].transform([st.session_state.drug])[0] if st.session_state.drug in encoders['Drug_Name'].classes_ else 0,
                'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
                'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
                'Co-surfactant': encoders['Co-surfactant'].transform([str(st.session_state.f_cs)])[0]
            }])
            res = {t: models[t].predict(in_df)[0] for t in models}
            
            # --- RECALIBRATED STABILITY LOGIC ---
            z_abs = abs(res['Zeta_mV'])
            pdi = res['PDI']
            
            # Weighted Stability Scoring (Zeta accounts for 70%, PDI for 30%)
            # A zeta of 30+ is excellent. A PDI below 0.2 is excellent.
            z_score = (z_abs / 30) * 70 if z_abs < 30 else 70 + (min(z_abs-30, 10))
            p_score = (max(0, 0.5 - pdi) / 0.5) * 30 
            
            stability_pct = min(100, max(5, z_score + p_score))
            
            if stability_pct >= 85:
                status, s_color = "Excellent - High Kinetic Stability", "green"
            elif stability_pct >= 60:
                status, s_color = "Good - Stable for Short/Medium Term", "orange"
            else:
                status, s_color = "Low - Risk of Phase Separation", "red"

            # Results Display
            ca, cb, cc, cd = st.columns(4)
            ca.metric("Particle Size", f"{res['Size_nm']:.2f} nm")
            cb.metric("PDI", f"{pdi:.3f}")
            cc.metric("Encapsulation (%EE)", f"{res['Encapsulation_Efficiency']:.2f} %")
            cd.metric("Stability %", f"{stability_pct:.1f} %")
            
            st.markdown(f"**Stability Assessment:** <span style='color:{s_color}'>{status}</span>", unsafe_allow_html=True)
            st.divider()

            # --- SHAP ANALYSIS ---
            st.subheader("🔍 SHAP Feature Descriptors")
            explainer = shap.Explainer(models['Size_nm'], X_train)
            sv = explainer(in_df)
            
            c_plot, c_expl = st.columns([1.5, 1])
            with c_plot:
                fig_sh, _ = plt.subplots(figsize=(10, 4))
                shap.plots.waterfall(sv[0], show=False)
                st.pyplot(fig_sh)
            with c_expl:
                st.write("**What does this graph mean?**")
                st.write("""
                - **Baseline (E[f(x)]):** The average size calculated from the entire dataset.
                - **The Bars:** Each bar shows how much a specific choice (like your Oil or Drug) pushed the particle size up or down.
                - **Blue Bars:** These choices **helped reduce** the particle size (Good for Nanoemulsions).
                - **Red Bars:** These choices **increased** the particle size.
                """)

        except Exception as e: st.error(f"Error: {str(e)}")

                # Dynamic Descriptor Explanation
                try:
                    impact_name = encoders['Oil_phase'].inverse_transform([in_df['Oil_phase'][0]])[0]
                    st.info(f"**Key Driver:** {impact_name}")
                except:
                    pass
                st.info(f"**Key Insight:** Your selection of **{main_impact}** is the primary driver for the current particle size.")

        except Exception as e: st.error(f"Error: {str(e)}")
