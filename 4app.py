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
        'Name of Drug': 'Drug_Name',
        'Name of Oil': 'Oil_phase',
        'Name of Surfactant': 'Surfactant',
        'Name of Cosurfactant': 'Co-surfactant',
        'Particle Size (nm)': 'Size_nm',
        'PDI': 'PDI',
        'Zeta Potential (mV)': 'Zeta_mV',
        '%EE': 'Encapsulation_Efficiency',
        'Method Used': 'Method' 
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
        else:
            df[col] = 0.0

    cat_cols = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant', 'Method']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['Not Stated', 'nan', 'None'], 'Unknown')
        else:
            df[col] = 'Unknown'

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

if df is not None:
    models, encoders, X_train, method_ai = train_models(df)

# --- STEP 1: SOURCING (WITH SMILES LOGIC) ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Chemical Profile & Smart Sourcing")
    uploaded_file = st.file_uploader("Industrial Work: Browse CSV File", type="csv")
    if uploaded_file: df = load_and_clean_data(uploaded_file)
    
    if df is not None:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.subheader("🧪 Chemical Input")
            drug = st.selectbox("Select Reference Drug", sorted(df['Drug_Name'].unique()))
            st.session_state.drug = drug
            
            smiles = st.text_input("Input Drug SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O")
            if smiles and RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Molecular Structure Identified")
                    logp = Descriptors.MolLogP(mol)
                    mw = Descriptors.MolWt(mol)
                    st.session_state.current_profile = {"logp": logp, "mw": mw}
                    st.write(f"**Calculated LogP:** {logp:.2f}")
                    st.write(f"**Mol Weight:** {mw:.2f} g/mol")
                else:
                    st.error("Invalid SMILES format.")

        with c2:
            st.subheader("🎯 Matched Production Components")
            if 'current_profile' in st.session_state:
                p = st.session_state.current_profile
                # Logic: Suggesting components based on Lipophilicity profile
                if p['logp'] > 3.5:
                    s_oil = [o for o in df['Oil_phase'].unique() if any(x in o.lower() for x in ['oleic', 'olive', 'corn', 'soy'])]
                    s_surf = [s for s in df['Surfactant'].unique() if '80' in s]
                else:
                    s_oil = [o for o in df['Oil_phase'].unique() if any(x in o.lower() for x in ['capryl', 'labra', 'miglyol'])]
                    s_surf = [s for s in df['Surfactant'].unique() if '20' in s or 'lecithin' in s.lower()]
                
                o_final = s_oil if s_oil else list(df['Oil_phase'].unique()[:3])
                s_final = s_surf if s_surf else list(df['Surfactant'].unique()[:3])
                cs_final = list(df['Co-surfactant'].unique()[:3])

                st.session_state.update({"o_matched": o_final, "s_matched": s_final, "cs_matched": cs_final})
                
                col1, col2, col3 = st.columns(3)
                col1.info("🛢️ **Optimal Oils**\n" + "\n".join([f"• {x}" for x in o_final[:3]]))
                col2.success("🧼 **Surfactants**\n" + "\n".join([f"• {x}" for x in s_final[:3]]))
                col3.warning("🧪 **Co-Surfactants**\n" + "\n".join([f"• {x}" for x in cs_final[:3]]))

        if st.button("Next: Solubility Analysis ➡️"):
            st.session_state.nav_index = 1
            st.rerun()

# --- STEP 2: SOLUBILITY (REACTIVE TO SMILES) ---
elif nav == "Step 2: Solubility":
    st.header("2. Reactive Solubility Profile")
    if 'drug' not in st.session_state: st.warning("Please go back to Step 1")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🔧 Component Selection")
            sel_o = st.selectbox("Oil Phase", st.session_state.get('o_matched', []) + list(df['Oil_phase'].unique()))
            sel_s = st.selectbox("Surfactant", st.session_state.get('s_matched', []) + list(df['Surfactant'].unique()))
            sel_cs = st.selectbox("Co-Surfactant", st.session_state.get('cs_matched', []) + list(df['Co-surfactant'].unique()))
            st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_cs": sel_cs})
        with c2:
            st.subheader("📊 Predicted Solubility")
            logp_val = st.session_state.get('current_profile', {'logp': 2.0})['logp']
            # Solubility estimate tied to SMILES LogP
            oil_sol = (logp_val * 0.8) + np.random.uniform(0.1, 0.5)
            st.metric(f"Solubility in {sel_o}", f"{oil_sol:.2f} mg/mL")
            st.metric(f"Solubility in {sel_s}", f"{(6 - logp_val) * 0.3:.2f} mg/mL")

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
        shift = (len(st.session_state.get('f_o', '')) + len(st.session_state.get('f_s', ''))) % 10
        za, zb = [5+shift, 15+shift, 25+shift, 5+shift], [40+shift, 60-shift, 40+shift, 40+shift]
        zc = [100 - a - b for a, b in zip(za, zb)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterternary(mode='markers', a=[oil], b=[smix], c=[100-oil-smix], marker=dict(size=15, color='red'), name="Selected Point"))
        fig.add_trace(go.Scatterternary(mode='lines', a=za, b=zb, c=zc, fill='toself', fillcolor='rgba(0,255,0,0.2)', line=dict(color='green'), name="Safe Zone"))
        fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
        st.plotly_chart(fig, use_container_width=True)
    if st.button("Next: AI Prediction ➡️"):
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4: PREDICTION ---
elif nav == "Step 4: AI Prediction":
    st.header("4. Batch Estimation & Interpretability")
    if 'f_o' not in st.session_state: st.warning("Please complete Step 2")
    else:
        try:
            in_df = pd.DataFrame([{
                'Drug_Name': encoders['Drug_Name'].transform([st.session_state.drug])[0],
                'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
                'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
                'Co-surfactant': encoders['Co-surfactant'].transform([str(st.session_state.f_cs)])[0]
            }])
            
            res = {t: models[t].predict(in_df)[0] for t in models}
            meth_idx = method_ai.predict(in_df)[0]
            meth_name = encoders['Method'].inverse_transform([meth_idx])[0]
            
            c_a, c_b, c_c = st.columns(3)
            c_a.metric("Size", f"{res['Size_nm']:.2f} nm"); c_a.metric("EE %", f"{res['Encapsulation_Efficiency']:.2f} %")
            c_b.metric("PDI", f"{res['PDI']:.3f}"); c_b.metric("Stability Score", f"{min(100, (abs(res['Zeta_mV'])/30)*100):.1f}/100")
            c_c.metric("Zeta", f"{res['Zeta_mV']:.2f} mV"); c_c.subheader("🛠️ Appropriate Method"); c_c.success(meth_name)
            
            st.divider()
            st.subheader("AI Decision Logic: SHAP Analysis Description")
            st.markdown("SHAP breaks down the prediction to show how much each component contributed to the final result.")
            explainer = shap.Explainer(models['Size_nm'], X_train)
            sv = explainer(in_df)
            fig_sh, _ = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(sv[0], show=False)
            st.pyplot(fig_sh)
        except Exception as e: st.error(f"Error: {str(e)}")
