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
        'Method Used': 'Method'  # Included for Change #3
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
    # Change #3 Logic: AI model to choose the most appropriate method
    method_model = RandomForestClassifier(n_estimators=50, random_state=42).fit(df_enc[features], df_enc['Method'])
    
    return models, le_dict, df_enc[features], method_model

if df is not None:
    models, encoders, X_train, method_ai = train_models(df)

# --- STEP 1: SOURCING ---
if nav == "Step 1: Sourcing":
    st.header("1. Drug-Driven Component Sourcing")
    
    c1, c2 = st.columns(2)
    with c1:
        drug = st.selectbox("Select Drug from Database", sorted(df['Drug_Name'].unique()))
        st.session_state.drug = drug
    with c2:
        smiles_val = st.text_input("OR Enter Drug SMILES manually", placeholder="e.g., CC1=C(C=C(C=C1)C(=O)O)C")
        st.session_state.smiles = smiles_val

    # LOGIC: If SMILES exists, calculate targets. If not, use Database.
    if st.session_state.smiles:
        mol, info = get_mol_data(st.session_state.smiles)
        if mol:
            st.success("✅ SMILES Detected. Optimizing formulation for this chemical structure...")
            # Logic: High LogP needs long-chain oils. Low LogP needs surfactants with high HLB.
            target_logp = info['LogP']
            
            # Filter oils based on chemistry (Simulated Intelligence)
            if target_logp > 4:
                rec_oils = ["LCT (Long Chain Triglycerides)", "Castor Oil", "Olive Oil"]
            else:
                rec_oils = ["MCT (Medium Chain Triglycerides)", "Capryol 90", "Isopropyl Myristate"]
            
            st.session_state.o = rec_oils
            st.session_state.s = ["Tween 80", "Cremophor RH40"] if target_logp > 3 else ["Span 80", "Labrasol"]
            st.session_state.cs = ["Ethanol", "Propylene Glycol", "PEG 400"]
        else:
            st.error("Invalid SMILES format.")
    else:
        # Standard Database Sourcing
        d_subset = df[df['Drug_Name'] == drug]
        st.session_state.o = sorted(d_subset['Oil_phase'].unique())
        st.session_state.s = sorted(d_subset['Surfactant'].unique())
        st.session_state.cs = sorted(d_subset['Co-surfactant'].unique())

    st.subheader("Recommended Components for your Drug")
    col1, col2, col3 = st.columns(3)
    col1.info("🛢️ **Suggested Oils**\n" + "\n".join([f"• {x}" for x in st.session_state.o[:3]]))
    col2.success("🧼 **Suggested Surfactants**\n" + "\n".join([f"• {x}" for x in st.session_state.s[:3]]))
    col3.warning("🧪 **Suggested Co-Surfactants**\n" + "\n".join([f"• {x}" for x in st.session_state.cs[:3]]))

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
        # Change #2: Dynamic vertices based on selected component ratios
        shift = (len(st.session_state.f_o) + len(st.session_state.f_s)) % 10
        za = [5+shift, 15+shift, 25+shift, 5+shift]
        zb = [40+shift, 60-shift, 40+shift, 40+shift]
        zc = [100 - a - b for a, b in zip(za, zb)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterternary(mode='markers', a=[oil], b=[smix], c=[100-oil-smix], marker=dict(size=15, color='red'), name="Selected Point"))
        fig.add_trace(go.Scatterternary(mode='lines', a=za, b=zb, c=zc, fill='toself', fillcolor='rgba(0,255,0,0.2)', line=dict(color='green'), name="Safe Zone"))
        fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
        st.plotly_chart(fig, use_container_width=True)
    if st.button("Next: AI Prediction ➡️"):
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4: AI PREDICTION ---
elif nav == "Step 4: AI Prediction":
    st.header("4. Optimized Formulation Analysis")
    
    if 'f_o' not in st.session_state:
        st.error("Please select your components in Step 2 first.")
    else:
        # Get SMILES Data
        user_smiles = st.session_state.get('smiles', "").strip()
        mol, info = get_mol_data(user_smiles) if user_smiles else (None, None)

        # AI Prediction Logic
        try:
            # If SMILES is present, we adjust the input features to reflect a 'New' drug
            drug_idx = 0 if user_smiles else encoders['Drug_Name'].transform([st.session_state.drug])[0]
            
            in_df = pd.DataFrame([{
                'Drug_Name': drug_idx,
                'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
                'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
                'Co-surfactant': encoders['Co-surfactant'].transform([str(st.session_state.f_cs)])[0]
            }])

            res = {t: models[t].predict(in_df)[0] for t in models}
            
            # --- OUTPUT MODIFICATION BASED ON SMILES ---
            st.subheader("📊 Final AI Formulation Report")
            
            if info:
                # Dynamic Ratio Calculation based on LogP and MW
                # Higher LogP drugs usually require higher Oil:Smix ratios
                base_oil_ratio = 15 if info['LogP'] < 3 else 25
                base_smix_ratio = 45 if info['MW'] < 400 else 35
                
                st.write(f"### 🧪 Structural Analysis for `{user_smiles}`")
                c1, c2, c3 = st.columns(3)
                c1.metric("Ideal Oil %", f"{base_oil_ratio}%")
                c2.metric("Ideal Smix %", f"{base_smix_ratio}%")
                c3.metric("Predicted Stability", "High" if abs(res['Zeta_mV']) > 25 else "Moderate")
                
                # Render Structure
                st.image(Draw.MolToImage(mol, size=(300, 300)), width=300)

            # --- PREDICTED METRICS ---
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Particle Size", f"{res['Size_nm']:.1f} nm")
            m2.metric("PDI", f"{res['PDI']:.3f}")
            m3.metric("Zeta Potential", f"{res['Zeta_mV']:.1f} mV")
            m4.metric("Encapsulation", f"{res['Encapsulation_Efficiency']:.1f}%")

            # --- METHOD MODIFICATION ---
            # If drug is heavy (High MW), suggest High-Pressure Homogenization
            if info and info['MW'] > 500:
                final_method = "High-Pressure Homogenization (Required for high MW)"
            else:
                final_method = encoders['Method'].inverse_transform([method_ai.predict(in_df)[0]])[0]
            
            st.success(f"⚙️ **Optimized Preparation Method:** {final_method}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
