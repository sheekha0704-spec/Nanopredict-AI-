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
            df[col] = df[col].apply(to_float).fillna(df[col].apply(to_float).median())
        else:
            df[col] = 0.0

    cat_cols = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant', 'Method']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['Not Stated', 'nan', 'None', 'Unknown'], 'Unknown')
        else:
            df[col] = 'Unknown'

    return df.dropna(subset=['Drug_Name', 'Oil_phase', 'Surfactant'])

# --- APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro", layout="wide")

if 'nav_index' not in st.session_state: st.session_state.nav_index = 0
steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

if 'uploaded_df' not in st.session_state: st.session_state.uploaded_df = None

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

df = load_and_clean_data(st.session_state.uploaded_df)
if df is not None:
    models, encoders, X_train, method_ai = train_models(df)

# --- STEP 1: SOURCING ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Formulation Sourcing & Profile")
    
    input_mode = st.radio("Select Sourcing Method:", 
                         ["Choose Drug from Database", "Browse File (Custom Data)", "Choose SMILES Profile"], 
                         horizontal=True)
    st.divider()
    
    c1, c2 = st.columns([1, 1.5])
    
    def clean_list(items):
        return sorted(list(set([x for x in items if str(x).lower() not in ['unknown', 'nan', 'none', 'not stated']])))

    with c1:
        current_profile = {"logp": 2.5, "mw": 200, "hbd": 1} 

        if input_mode == "Choose Drug from Database":
            st.subheader("📁 Database Selection")
            drug_choice = st.selectbox("Select Drug Name", sorted(df['Drug_Name'].unique()) if df is not None else ["No Data"])
            st.session_state.drug = drug_choice
            
            # PERSONALIZATION LOGIC: Extract components associated with THIS drug in the dataset
            d_subset = df[df['Drug_Name'] == drug_choice]
            o_rec = clean_list(d_subset['Oil_phase'].unique())
            s_rec = clean_list(d_subset['Surfactant'].unique())
            cs_rec = clean_list(d_subset['Co-surfactant'].unique())
            
            # Infer LogP based on the drug's common formulation partners
            if any(o in str(d_subset['Oil_phase'].values).lower() for o in ['oleic', 'soy', 'olive']):
                current_profile['logp'] = 4.5
            else:
                current_profile['logp'] = 2.1
            st.info(f"Database Record: {drug_choice}")

        elif input_mode == "Browse File (Custom Data)":
            st.subheader("📤 Industrial Upload")
            up_file = st.file_uploader("Upload Lab Results (CSV)", type="csv")
            if up_file:
                st.session_state.uploaded_df = up_file
                st.rerun()
            o_rec, s_rec, cs_rec = [], [], []

        elif input_mode == "Choose SMILES Profile":
            st.subheader("🧪 SMILES Analysis")
            smiles = st.text_input("Input Drug SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O")
            if smiles and RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Molecular Architecture")
                    current_profile['logp'] = Descriptors.MolLogP(mol)
                    current_profile['mw'] = Descriptors.MolWt(mol)
                    st.write(f"**Calculated LogP:** {current_profile['logp']:.2f}")
                    
                    # CHEMICAL MATCHING LOGIC
                    if current_profile['logp'] > 3.0:
                        o_rec = clean_list([o for o in df['Oil_phase'].unique() if any(x in o.lower() for x in ['oleic', 'soy', 'olive', 'lct'])])
                        s_rec = clean_list([s for s in df['Surfactant'].unique() if '80' in s or 'lecithin' in s.lower()])
                    else:
                        o_rec = clean_list([o for o in df['Oil_phase'].unique() if any(x in o.lower() for x in ['capryl', 'labra', 'miglyol', 'mct'])])
                        s_rec = clean_list([s for s in df['Surfactant'].unique() if '20' in s or 'solutol' in s.lower()])
                    cs_rec = clean_list(df['Co-surfactant'].unique())
                else: st.error("Invalid SMILES.")
            st.session_state.drug = df['Drug_Name'].iloc[0] if df is not None else "Unknown"

        st.session_state.current_profile = current_profile

    with c2:
        st.subheader("🎯 Personalised Recommendations")
        if df is not None:
            # Display matched components
            cola, colb, colc = st.columns(3)
            with cola: 
                st.success("🛢️ Best Oils")
                for x in o_rec[:5] if o_rec else clean_list(df['Oil_phase'].unique()[:3]): st.write(f"- {x}")
            with colb: 
                st.success("🧼 Best Surfactants")
                for x in s_rec[:5] if s_rec else clean_list(df['Surfactant'].unique()[:3]): st.write(f"- {x}")
            with colc: 
                st.success("🧪 Best Co-Surfactants")
                for x in cs_rec[:5] if cs_rec else clean_list(df['Co-surfactant'].unique()[:3]): st.write(f"- {x}")
            
            st.session_state.update({"o_matched": o_rec if o_rec else list(df['Oil_phase'].unique()), 
                                    "s_matched": s_rec if s_rec else list(df['Surfactant'].unique()), 
                                    "cs_matched": cs_rec if cs_rec else list(df['Co-surfactant'].unique())})

    if st.button("Next: Solubility Analysis ➡️"):
        st.session_state.nav_index = 1
        st.rerun()

# --- STEP 2: SOLUBILITY ---
elif nav == "Step 2: Solubility":
    st.header("2. Predicted Solubility Profile")
    if 'current_profile' not in st.session_state: st.warning("Please complete Step 1.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            sel_o = st.selectbox("Oil Phase", st.session_state.o_matched + ["--- OTHERS ---"] + list(df['Oil_phase'].unique()))
            sel_s = st.selectbox("Surfactant", st.session_state.s_matched + ["--- OTHERS ---"] + list(df['Surfactant'].unique()))
            sel_cs = st.selectbox("Co-surfactant", st.session_state.cs_matched + ["--- OTHERS ---"] + list(df['Co-surfactant'].unique()))
            st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_cs": sel_cs})
        with c2:
            logp_val = st.session_state.current_profile['logp']
            oil_sol = (logp_val * 0.8) + 1.2
            st.metric(f"Solubility in {sel_o}", f"{max(0.1, oil_sol):.2f} mg/mL")

        if st.button("Next: Ternary Mapping ➡️"):
            st.session_state.nav_index = 2
            st.rerun()

# --- STEP 3: TERNARY ---
elif nav == "Step 3: Ternary":
    st.header("3. Ternary Phase Optimization")
    l, r = st.columns([1, 2])
    with l:
        smix, oil = st.slider("Smix %", 10, 80, 40), st.slider("Oil %", 5, 40, 15)
    with r:
        shift = (len(st.session_state.get('f_o', '')) + len(st.session_state.get('f_s', ''))) % 10
        za, zb = [5+shift, 15+shift, 25+shift, 5+shift], [40+shift, 60-shift, 40+shift, 40+shift]
        zc = [100 - a - b for a, b in zip(za, zb)]
        fig = go.Figure()
        fig.add_trace(go.Scatterternary(mode='markers', a=[oil], b=[smix], c=[100-oil-smix], marker=dict(size=15, color='red')))
        fig.add_trace(go.Scatterternary(mode='lines', a=za, b=zb, c=zc, fill='toself', fillcolor='rgba(0,255,0,0.2)', line=dict(color='green')))
        fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
        st.plotly_chart(fig, use_container_width=True)
    if st.button("Next: AI Prediction ➡️"):
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4: PREDICTION ---
elif nav == "Step 4: AI Prediction":
    st.header("4. AI Batch Estimation & Explainability")
    if 'f_o' not in st.session_state: st.warning("Please complete steps.")
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
            st.subheader("🔍 SHAP Feature Descriptors")
            explainer = shap.Explainer(models['Size_nm'], X_train)
            sv = explainer(in_df)
            
            # TEXT-BASED SHAP DESCRIPTORS
            cols = st.columns(4)
            features = ["Drug Selection", "Oil Phase", "Surfactant", "Co-surfactant"]
            for i, feat in enumerate(features):
                impact = sv.values[0][i]
                direction = "📈 Increases" if impact > 0 else "📉 Decreases"
                color = "red" if impact > 0 else "green"
                cols[i].markdown(f"**{feat}**")
                cols[i].markdown(f"<span style='color:{color}'>{direction} size by {abs(impact):.2f} nm</span>", unsafe_allow_html=True)

            st.write("### AI Decision Logic (Waterfall Plot)")
            fig_sh, _ = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(sv[0], show=False)
            st.pyplot(fig_sh)
            
        except Exception as e: st.error(f"Error: {str(e)}")
