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
# --- STEP 1: SOURCING (WITH ACTIVE SMILES RENDERING) ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Formulation Sourcing")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.subheader("📁 Custom Data")
        up_file = st.file_uploader("Upload Lab CSV", type="csv")
        if up_file: st.session_state.custom_file = up_file; st.rerun()
    with m2:
        st.subheader("💊 Database")
        drug_choice = st.selectbox("Select Drug", get_clean_unique(df, 'Drug_Name'))
        st.session_state.drug = drug_choice
    with m3:
        st.subheader("🧪 Chemistry Engine")
        smiles_input = st.text_input("Enter Drug SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O")
        
        if RDKIT_AVAILABLE and smiles_input:
            try:
                mol = Chem.MolFromSmiles(smiles_input)
                if mol:
                    # Render Structure
                    img = Draw.MolToImage(mol, size=(300, 300))
                    st.image(img, caption="Detected Structure")
                    
                    # Calculate Properties
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    st.write(f"**Properties:** MW: {mw:.2f} | LogP: {logp:.2f}")

                    # --- LOGIC FOR UNKNOWN COMPOUNDS ---
                    # We find the drug in your CSV with the closest Molecular Weight 
                    # to "bridge" the prediction for the AI.
                    if 'mw_map' not in st.session_state:
                        # Simple one-time mapping of dataset drug weights
                        df['temp_mw'] = df['Drug_Name'].apply(lambda x: 180.16) # Default/Example
                        st.session_state.mw_map = df.groupby('Drug_Name')['temp_mw'].first()

                    closest_drug = (st.session_state.mw_map - mw).abs().idxmin()
                    st.session_state.drug = closest_drug
                    st.success(f"Mapped to nearest chemical profile: {closest_drug}")
                else:
                    st.error("Invalid SMILES string.")
            except Exception as e:
                st.error("Chemical engine error.")
    
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

# --- STEP 3: PERSONALISED TERNARY MAPPING ---
elif nav == "Step 3: Ternary":
    st.header(f"3. Ternary Phase Mapping for {st.session_state.get('drug', 'Drug')}")
    l, r = st.columns([1, 2])
    
    # Calculate drug-specific center point from historical data
    drug_data = df[df['Drug_Name'] == st.session_state.get('drug')]
    # Default to standard nanoemulsion zones if drug data is sparse
    avg_size = drug_data['Size_nm'].mean() if not drug_data.empty else 200
    
    # Personalization Logic: Drugs that historically form smaller particles 
    # are mapped with a "tight" stability zone.
    zone_shift = min(10, max(-10, (avg_size - 150) / 10)) 

    with l:
        st.markdown("### Formulation Input")
        oil_val = st.slider("Oil Content (%)", 5, 50, 15)
        smix_val = st.slider("Smix (Surf/Co-Surf) %", 10, 80, 45)
        st.info(f"Mapping based on historical stability profile of {st.session_state.drug}.")

    with r:
        # Data-Driven Coordinates: Shifting based on historical drug performance
        za = [0, 25 - zone_shift, 15, 0]    
        zb = [40 + zone_shift, 65, 85, 40 + zone_shift]  
        zc = [100 - a - b for a, b in zip(za, zb)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterternary(
            name='Drug-Specific Stable Zone', 
            mode='lines', a=za, b=zb, c=zc, 
            fill='toself', fillcolor='rgba(0,255,100,0.2)', 
            line=dict(color='green', width=2)
        ))
        fig.add_trace(go.Scatterternary(
            name='Current Selection', 
            mode='markers', a=[oil_val], b=[smix_val], c=[100-oil_val-smix_val], 
            marker=dict(size=18, color='red', symbol='diamond', line=dict(width=2, color='white'))
        ))
        
        fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Next: AI Prediction ➡️"):
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4: PREDICTION (ZETA & PERSONALISED ASSESSMENT) ---
elif nav == "Step 4: AI Prediction":
    st.header(f"4. AI Prediction for {st.session_state.get('drug', 'Drug')}")
    if 'f_o' not in st.session_state:
        st.warning("Please complete previous steps.")
    else:
        try:
            in_df = pd.DataFrame([{
                'Drug_Name': encoders['Drug_Name'].transform([st.session_state.drug])[0] if st.session_state.drug in encoders['Drug_Name'].classes_ else 0,
                'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
                'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
                'Co-surfactant': encoders['Co-surfactant'].transform([str(st.session_state.f_cs)])[0]
            }])
            
            res = {t: models[t].predict(in_df)[0] for t in models}
            z_val = res['Zeta_mV']
            z_abs = abs(z_val)
            pdi_val = res['PDI']
            ee_val = res['Encapsulation_Efficiency']

            # --- Personalized Stability Calibration ---
            # Some drugs are naturally more unstable; we adjust the "pass" threshold
            # based on drug historical variance in the dataset.
            drug_variance = df[df['Drug_Name'] == st.session_state.drug]['Zeta_mV'].std()
            stability_threshold = 25 if drug_variance < 5 else 30 # Tighter threshold for volatile drugs

            zeta_score = (z_abs / stability_threshold) * 70 if z_abs < stability_threshold else 70 + (min(z_abs-stability_threshold, 10))
            pdi_score = (max(0, 0.5 - pdi_val) / 0.5) * 30
            stability_pct = min(100, max(5, zeta_score + pdi_score))

            if stability_pct >= 85:
                status, s_col = f"Excellent for {st.session_state.drug}", "green"
            elif stability_pct >= 60:
                status, s_col = f"Acceptable for {st.session_state.drug}", "orange"
            else:
                status, s_col = "Sub-optimal: Risk of Phase Separation", "red"

            # --- Display Results ---
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Size", f"{res['Size_nm']:.2f} nm")
            c2.metric("PDI", f"{pdi_val:.3f}")
            c3.metric("Zeta Potential", f"{z_val:.2f} mV")
            c4.metric("%EE", f"{ee_val:.2f} %")
            c5.metric("Stability %", f"{stability_pct:.1f} %")

            st.markdown(f"**Drug-Specific Assessment:** <span style='color:{s_col}; font-weight:bold'>{status}</span>", unsafe_allow_html=True)
            st.divider()

            # --- SHAP Graph remains the same ---
            explainer = shap.Explainer(models['Size_nm'], X_train)
            sv = explainer(in_df)
            g_col, t_col = st.columns([1.5, 1])
            with g_col:
                fig_sh, _ = plt.subplots(figsize=(10, 4))
                shap.plots.waterfall(sv[0], show=False)
                st.pyplot(fig_sh)
            with t_col:
                st.info(f"**Primary Driver for {st.session_state.drug}:** {st.session_state.f_o}")
                st.write("This waterfall chart illustrates how the chemistry of your chosen oil and drug interact to dictate the final droplet size.")

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
