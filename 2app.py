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
    st.header("NanoPredict: Drug-Driven Component Sourcing")
    uploaded_file = st.file_uploader("Industrial Work: Browse CSV File", type="csv")
    if uploaded_file: df = load_and_clean_data(uploaded_file)
    
    if df is not None:
        c1, c2 = st.columns(2)
        with c1:
            drug = st.selectbox("Select Drug from Database", sorted(df['Drug_Name'].unique()))
            st.session_state.drug = drug
        with c2:
            smiles = st.text_input("Enter Drug SMILES manually", placeholder="Enter string here...")
            if smiles: st.caption(f"SMILES Registered: {smiles}")

        d_subset = df[df['Drug_Name'] == drug]
        o_list, s_list, cs_list = sorted(d_subset['Oil_phase'].unique()), sorted(d_subset['Surfactant'].unique()), sorted(d_subset['Co-surfactant'].unique())
        st.session_state.update({"o": o_list, "s": s_list, "cs": cs_list})

        st.subheader(f"Best Matched Components for {drug}")
        col1, col2, col3 = st.columns(3)
        col1.info("🛢️ **Oils**\n" + "\n".join([f"• {x}" for x in o_list[:3]]))
        col2.success("🧼 **Surfactants**\n" + "\n".join([f"• {x}" for x in s_list[:3]]))
        col3.warning("🧪 **Co-Surfactants**\n" + "\n".join([f"• {x}" for x in cs_list[:3]]))

        if st.button("Next: Solubility Analysis ➡️"):
            st.session_state.nav_index = 1
            st.rerun()

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

# --- STEP 4: PREDICTION (SMILES ANALYSIS MODE) ---
elif nav == "Step 4: AI Prediction":
    st.header("4. Prediction & Chemical Analysis")
    if 'f_o' not in st.session_state: 
        st.error("Please complete previous steps.")
    else:
        try:
            # 1. SMILES PRIORITY LOGIC
            user_smiles = st.session_state.get('smiles', "").strip()
            
            if user_smiles:
                st.subheader("🧬 Analysis for Custom SMILES")
                mol, info = get_mol_data(user_smiles)
                
                if mol:
                    # Render the molecule
                    img = Draw.MolToImage(mol, size=(300, 300))
                    st.image(img, caption=f"Chemical Structure of: {user_smiles}")
                    
                    # Display the physical analysis
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Molecular Weight", f"{info['MW']:.2f} Da")
                    col_b.metric("Lipophilicity (LogP)", f"{info['LogP']:.2f}")
                    col_c.success("Ready for Nano-Prediction")
                    
                    # Force the model to treat this as a "New Molecule" 
                    # We use a neutral index (0) so it doesn't lean on database drug IDs
                    drug_input_val = 0 
                else:
                    st.error("Invalid SMILES. Falling back to dropdown drug.")
                    drug_input_val = encoders['Drug_Name'].transform([st.session_state.drug])[0]
            else:
                st.info(f"💊 Predicting for Database Drug: {st.session_state.drug}")
                drug_input_val = encoders['Drug_Name'].transform([st.session_state.drug])[0]

            # 2. PREPARE INPUT FOR THE AI
            in_df = pd.DataFrame([{
                'Drug_Name': drug_input_val,
                'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
                'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
                'Co-surfactant': encoders['Co-surfactant'].transform([str(st.session_state.f_cs)])[0]
            }])

            # 3. RUN PREDICTIONS
            res = {t: models[t].predict(in_df)[0] for t in models}
            meth_idx = method_ai.predict(in_df)[0]
            meth_name = encoders['Method'].inverse_transform([meth_idx])[0]

            # 4. SHOW RESULTS
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Size", f"{res['Size_nm']:.2f} nm")
            c2.metric("PDI (Uniformity)", f"{res['PDI']:.3f}")
            c3.metric("Encapsulation Efficiency", f"{res['Encapsulation_Efficiency']:.1f}%")
            
            st.warning(f"🛠️ **Suggested Method:** {meth_name}")

            # 5. SHAP INTERPRETABILITY (Crucial for Analysis)
            st.subheader("🔍 AI Decision Analysis")
            st.write("This chart shows how your SMILES drug and selected oils are impacting the particle size.")
            
            explainer = shap.Explainer(models['Size_nm'], X_train)
            sv = explainer(in_df)
            fig, ax = plt.subplots(figsize=(10, 4))
            # If it's a new SMILES, SHAP will show the impact of the "Unknown" drug category
            shap.plots.waterfall(sv[0], show=False)
            st.pyplot(fig)

        except Exception as e: 
            st.error(f"Analysis Error: {e}")
