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

# --- 1. DATA ENGINE (MODIFIED FOR 3 FILES) ---
@st.cache_data
def load_and_clean_data(uploaded_file, default_path):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif os.path.exists(default_path):
        df = pd.read_csv(default_path)
    else:
        return None
    
    # Standardize column names across different potential file formats
    column_mapping = {
        'Name of Drug': 'Drug_Name', 'Name of Oil': 'Oil_phase',
        'Name of Surfactant': 'Surfactant', 'Name of Cosurfactant': 'Co-surfactant',
        'Particle Size (nm)': 'Size_nm', 'PDI': 'PDI',
        'Zeta Potential (mV)': 'Zeta_mV', '%EE': 'Encapsulation_Efficiency',
        'Method Used': 'Method', 'Solubility': 'Solubility_Value'
    }
    df = df.rename(columns=column_mapping)
    df.columns = [c.strip() for c in df.columns]
    return df

# Helper to clean lists
def clean_list(items):
    return sorted([x for x in items if str(x).lower() not in ['unknown', 'nan', 'none', 'not stated']])

# --- APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro", layout="wide")

if 'nav_index' not in st.session_state: st.session_state.nav_index = 0
steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

# File Management for 3-file system
st.sidebar.header("📁 Data Sourcing")
file1 = st.sidebar.file_uploader("Upload Sourcing Data (Step 1)", type="csv")
file2 = st.sidebar.file_uploader("Upload Solubility Data (Step 2)", type="csv")
file3 = st.sidebar.file_uploader("Upload Prediction Data (Step 4)", type="csv")

df1 = load_and_clean_data(file1, 'nanoemulsion_sourcing.csv')
df2 = load_and_clean_data(file2, 'nanoemulsion_solubility.csv')
df3 = load_and_clean_data(file3, 'nanoemulsion_main.csv')

# --- STEP 1: SOURCING ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Formulation Sourcing & Profile")
    
    input_mode = st.radio("Select Sourcing Method:", 
                         ["Choose Drug from Database", "Choose SMILES Profile"], 
                         horizontal=True)
    st.divider()
    
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        current_profile = {"logp": 2.5, "mw": 200, "hbd": 1} 
        if input_mode == "Choose Drug from Database" and df1 is not None:
            drug_choice = st.selectbox("Select Drug Name", sorted(df1['Drug_Name'].unique()))
            st.session_state.drug = drug_choice
            
            # CHANGE 1: Personalizing component matching based on Drug selection
            drug_data = df1[df1['Drug_Name'] == drug_choice].iloc[0]
            # If your CSV has LogP/MW per drug, use it, else fallback to heuristic
            current_profile['logp'] = drug_data.get('LogP', 2.5) 
            st.info(f"Database Record: {drug_choice}")

        elif input_mode == "Choose SMILES Profile":
            smiles = st.text_input("Input Drug SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O")
            if smiles and RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Molecular Structure")
                    current_profile = {"logp": Descriptors.MolLogP(mol), "mw": Descriptors.MolWt(mol), "hbd": Descriptors.NumHDonors(mol)}
                    st.write(f"**Calculated LogP:** {current_profile['logp']:.2f}")
            st.session_state.drug = "Custom SMILES"
        
        st.session_state.current_profile = current_profile

    with c2:
        st.subheader("🎯 Smart-Matched Components")
        if df1 is not None and 'drug' in st.session_state:
            # Filter components specifically linked to this drug in File 1
            drug_subset = df1[df1['Drug_Name'] == st.session_state.drug]
            
            o_final = clean_list(drug_subset['Oil_phase'].unique())
            s_final = clean_list(drug_subset['Surfactant'].unique())
            cs_final = clean_list(drug_subset['Co-surfactant'].unique())

            cola, colb, colc = st.columns(3)
            with cola: 
                st.success("🛢️ Best Oils")
                for x in o_final[:5]: st.write(f"- {x}")
            with colb: 
                st.success("🧼 Best Surfactants")
                for x in s_final[:5]: st.write(f"- {x}")
            with colc: 
                st.success("🧪 Best Co-Surfactants")
                for x in cs_final[:5]: st.write(f"- {x}")
            
            st.session_state.update({"o_matched": o_final, "s_matched": s_final, "cs_matched": cs_final})

    if st.button("Next: Solubility Analysis ➡️"):
        st.session_state.nav_index = 1
        st.rerun()

# --- STEP 2: SOLUBILITY ---
elif nav == "Step 2: Solubility":
    st.header("2. Predicted Solubility Profile")
    if 'current_profile' not in st.session_state: 
        st.warning("Please complete Step 1.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            # User picks from matched list or full database
            sel_o = st.selectbox("Select Oil", st.session_state.get('o_matched', []) + ["--- OTHERS ---"] + list(df2['Oil_phase'].unique() if df2 is not None else []))
            sel_s = st.selectbox("Select Surfactant", st.session_state.get('s_matched', []) + ["--- OTHERS ---"] + list(df2['Surfactant'].unique() if df2 is not None else []))
            sel_cs = st.selectbox("Select Co-Surfactant", st.session_state.get('cs_matched', []) + ["--- OTHERS ---"] + list(df2['Co-surfactant'].unique() if df2 is not None else []))
            st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_cs": sel_cs})

        with c2:
            # CHANGE 2: Solubility altering according to change in oil/surfactant
            # We look up values from df2 (the solubility file)
            def get_solubility(comp_name, comp_type):
                if df2 is not None:
                    match = df2[(df2['Drug_Name'] == st.session_state.drug) & (df2[comp_type] == comp_name)]
                    if not match.empty:
                        return match['Solubility_Value'].mean()
                # Fallback to calculation if not in DB
                logp = st.session_state.current_profile['logp']
                return (logp * 0.8) + 1.2 if comp_type == 'Oil_phase' else (6 - logp) * 0.3

            s_o = get_solubility(sel_o, 'Oil_phase')
            s_s = get_solubility(sel_s, 'Surfactant')
            s_cs = get_solubility(sel_cs, 'Co-surfactant')

            st.metric(f"Solubility in {sel_o}", f"{s_o:.2f} mg/mL")
            st.metric(f"Solubility in {sel_s}", f"{s_s:.2f} mg/mL")
            st.metric(f"Solubility in {sel_cs}", f"{s_cs:.2f} mg/mL")

        if st.button("Next: Ternary Mapping ➡️"):
            st.session_state.nav_index = 2
            st.rerun()

# --- STEP 3: TERNARY --- (Code remains as per your ternary logic)
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
    st.header("4. AI Batch Estimation & Logic")
    if df3 is None:
        st.error("Please upload the prediction dataset (File 3) in the sidebar.")
    elif 'f_o' not in st.session_state:
        st.warning("Please complete steps.")
    else:
        # Training logic moved here to ensure it uses df3 (the prediction file)
        @st.cache_resource
        def train_final_models(_data):
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

        models, encoders, X_train, method_ai = train_final_models(df3)

        try:
            in_df = pd.DataFrame([{
                'Drug_Name': encoders['Drug_Name'].transform([st.session_state.drug])[0],
                'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
                'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
                'Co-surfactant': encoders['Co-surfactant'].transform([str(st.session_state.f_cs)])[0]
            }])
            res = {t: models[t].predict(in_df)[0] for t in models}
            meth_name = encoders['Method'].inverse_transform([method_ai.predict(in_df)[0]])[0]
            
            ca, cb, cc = st.columns(3)
            with ca:
                st.metric("Size", f"{res['Size_nm']:.2f} nm")
                st.metric("EE %", f"{res['Encapsulation_Efficiency']:.2f} %")
            with cb:
                st.metric("PDI", f"{res['PDI']:.3f}")
                st.metric("Stability Score", f"{min(100, (abs(res['Zeta_mV'])/30)*100):.1f}/100")
            with cc:
                st.metric("Zeta", f"{res['Zeta_mV']:.2f} mV")
                st.success(f"Method: {meth_name}")
            
            st.divider()
            explainer = shap.Explainer(models['Size_nm'], X_train)
            sv = explainer(in_df)
            fig_sh, _ = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(sv[0], show=False)
            st.pyplot(fig_sh)
        except Exception as e: 
            st.error(f"Input mismatch: {str(e)}. Ensure the drug and components exist in the prediction file.")
