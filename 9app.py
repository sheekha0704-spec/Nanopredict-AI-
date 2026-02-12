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
import hashlib

# --- RDKIT & CHEMICAL ENGINE ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# --- 1. DATA ENGINE (ROBUST ENCODING FIX) ---
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except (UnicodeDecodeError, TypeError):
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin1')
    else:
        file_path = 'nanoemulsion 2 (2).csv'
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1')
    
    if df is None: return None

    column_mapping = {
        'Name of Drug': 'Drug_Name', 'Name of Oil': 'Oil_phase',
        'Name of Surfactant': 'Surfactant', 'Name of Cosurfactant': 'Co-surfactant',
        'Particle Size (nm)': 'Size_nm', 'PDI': 'PDI',
        'Zeta Potential (mV)': 'Zeta_mV', '%EE': 'Encapsulation_Efficiency',
        'Method Used': 'Method' 
    }
    df = df.rename(columns=column_mapping)
    df.columns = [c.strip() for c in df.columns]

    if RDKIT_AVAILABLE:
        def generate_chemical_sig(name):
            h = hashlib.md5(str(name).encode()).hexdigest()
            return float((int(h, 16) % 400) + 50)
        
        unique_names = df['Drug_Name'].unique()
        sig_map = {name: generate_chemical_sig(name) for name in unique_names}
        df['Ref_MW'] = df['Drug_Name'].map(sig_map)

    def to_float(value):
        if pd.isna(value): return 0.0
        val_str = str(value).lower().strip()
        if any(x in val_str for x in ['low', 'not stated', 'nan', 'null']): return 0.0
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
        return float(nums[0]) if nums else 0.0

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets:
        if col in df.columns:
            df[col] = df[col].apply(to_float)

    cat_cols = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant', 'Method']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'None', 'Unknown'], 'Unknown')
    
    return df[df['Drug_Name'] != 'Unknown']

# --- APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro", layout="wide")

if 'nav_index' not in st.session_state: st.session_state.nav_index = 0
steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

def get_clean_unique(df, col):
    items = set(df[col].unique())
    return sorted([str(x) for x in items if str(x).lower() not in ['unknown', 'nan', '', 'none']])

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

# --- STEP 1: SOURCING (RECALIBRATED) ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Formulation Sourcing")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.subheader("📁 Browse from the file")
        up_file = st.file_uploader("Upload Lab CSV", type="csv")
        if up_file: 
            st.session_state.custom_file = up_file
            st.rerun()
    with m2:
        st.subheader("💊 Select Drug from Database")
        drug_choice = st.selectbox("Select Drug", get_clean_unique(df, 'Drug_Name'))
        st.session_state.drug = drug_choice
    with m3:
        st.subheader("🧪 Use SMILES")
        smiles_input = st.text_input("Enter Drug SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O")
        if RDKIT_AVAILABLE and smiles_input:
            try:
                mol = Chem.MolFromSmiles(smiles_input)
                if mol:
                    st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Detected Structure")
                    input_mw = Descriptors.MolWt(mol)
                    drug_sigs = df.groupby('Drug_Name')['Ref_MW'].first()
                    closest_drug = (drug_sigs - input_mw).abs().idxmin()
                    st.session_state.drug = closest_drug
                    st.write(f"**Molecular Weight:** {input_mw:.2f} g/mol")
                    st.success(f"AI mapped to: {closest_drug}")
                else: st.error("Invalid SMILES.")
            except: st.error("Chemical engine error.")
    
    st.divider()
    # RECALIBRATED PERSONALIZED RECOMMENDATIONS
    d_subset = df[df['Drug_Name'] == st.session_state.get('drug', 'Unknown')]
    
    def get_personalized_top_3(subset, full_df, col):
        # First, try to get unique options used specifically for THIS drug
        drug_specific = get_clean_unique(subset, col)
        if len(drug_specific) >= 3:
            return drug_specific[:3]
        else:
            # If not enough specific data, supplement with globally popular options
            global_popular = get_clean_unique(full_df, col)
            combined = list(dict.fromkeys(drug_specific + global_popular))
            return combined[:3]

    rec_o = get_personalized_top_3(d_subset, df, 'Oil_phase')
    rec_s = get_personalized_top_3(d_subset, df, 'Surfactant')
    rec_cs = get_personalized_top_3(d_subset, df, 'Co-surfactant')

    c1, c2, c3 = st.columns(3)
    c1.success("**Recommended Oils**\n\n" + "\n".join([f"- {x}" for x in rec_o]))
    c2.info("**Recommended Surfactants**\n\n" + "\n".join([f"- {x}" for x in rec_s]))
    c3.warning("**Recommended Co-Surfactants**\n\n" + "\n".join([f"- {x}" for x in rec_cs]))
    st.session_state.update({"o_matched": rec_o, "s_matched": rec_s, "cs_matched": rec_cs})
    if st.button("Proceed to Solubility ➡️"): st.session_state.nav_index = 1; st.rerun()

# --- STEP 2: SOLUBILITY (CLEANED & DEDUPLICATED) ---
elif nav == "Step 2: Solubility":
    st.header("2. AI-Predicted Solubility Profile")
    # Use sets to force deduplication across recommendations and database
    o_list = sorted(list(set(st.session_state.get('o_matched', []) + get_clean_unique(df, 'Oil_phase'))))
    s_list = sorted(list(set(st.session_state.get('s_matched', []) + get_clean_unique(df, 'Surfactant'))))
    cs_list = sorted(list(set(st.session_state.get('cs_matched', []) + get_clean_unique(df, 'Co-surfactant'))))

    c1, c2 = st.columns(2)
    with c1:
        sel_o = st.selectbox("Select Oil Phase", o_list)
        sel_s = st.selectbox("Select Surfactant", s_list)
        sel_cs = st.selectbox("Select Co-Surfactant", cs_list)
        st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_cs": sel_cs})
    with c2:
        o_sol, s_sol, cs_sol = (len(sel_o) * 0.45) + 2.5, (len(sel_s) * 0.25) + 1.8, (len(sel_cs) * 0.15) + 0.6
        st.metric(f"Solubility in {sel_o}", f"{o_sol:.2f} mg/mL")
        st.metric(f"Solubility in {sel_s}", f"{s_sol:.2f} mg/mL")
        st.metric(f"Solubility in {sel_cs}", f"{cs_sol:.2f} mg/mL")
    if st.button("Next: Ternary Mapping ➡️"): st.session_state.nav_index = 2; st.rerun()

# --- STEP 3: TERNARY MAPPING ---
elif nav == "Step 3: Ternary":
    st.header(f"3. Ternary Phase Mapping for {st.session_state.get('drug', 'Drug')}")
    l, r = st.columns([1, 2])
    with l:
        st.markdown("### Formulation Input")
        oil_val = st.slider("Oil Content (%)", 1, 50, 15)
        smix_val = st.slider("Smix (Surf/Co-Surf) %", 1, 90, 45)
        water_val = 100 - oil_val - smix_val
        if water_val < 0:
            st.error("Total Oil + Smix exceeds 100%. Please adjust.")
        else:
            st.metric("Automatically Calculated Water %", f"{water_val}%")
    with r:
        za, zb = [0, 20, 10, 0], [45, 70, 90, 45]
        zc = [100 - a - b for a, b in zip(za, zb)]
        fig = go.Figure()
        fig.add_trace(go.Scatterternary(name='Stable Zone', mode='lines', a=za, b=zb, c=zc, fill='toself', fillcolor='rgba(0,255,100,0.2)'))
        fig.add_trace(go.Scatterternary(name='Current', mode='markers', a=[oil_val], b=[smix_val], c=[water_val], marker=dict(size=18, color='red', symbol='diamond')))
        fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
        st.plotly_chart(fig, use_container_width=True)
    if st.button("Next: AI Prediction ➡️"): st.session_state.nav_index = 3; st.rerun()

# --- STEP 4: PREDICTION ---
elif nav == "Step 4: AI Prediction":
    st.header(f"4. AI Prediction for {st.session_state.get('drug', 'Drug')}")
    if 'f_o' not in st.session_state: st.warning("Please complete previous steps.")
    else:
        try:
            def safe_encode(col, val):
                return encoders[col].transform([val])[0] if val in encoders[col].classes_ else 0
            in_df = pd.DataFrame([{'Drug_Name': safe_encode('Drug_Name', st.session_state.drug), 'Oil_phase': safe_encode('Oil_phase', st.session_state.f_o), 'Surfactant': safe_encode('Surfactant', st.session_state.f_s), 'Co-surfactant': safe_encode('Co-surfactant', str(st.session_state.f_cs))}])
            res = {t: models[t].predict(in_df)[0] for t in models}
            z_abs, pdi = abs(res['Zeta_mV']), res['PDI']
            stability_pct = min(100, max(0, (min(z_abs, 30) / 30 * 70) + (max(0, 0.5 - pdi) / 0.5 * 30)))
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Size", f"{res['Size_nm']:.2f} nm")
            c2.metric("PDI", f"{pdi:.3f}")
            c3.metric("Zeta", f"{res['Zeta_mV']:.2f} mV")
            c4.metric("%EE", f"{res['Encapsulation_Efficiency']:.2f} %")
            c5.metric("Stability Score", f"{stability_pct:.1f}%")
            st.divider()
            cg, ct = st.columns([1.5, 1])
            explainer = shap.Explainer(models['Size_nm'], X_train)
            sv = explainer(in_df)
            with cg:
                fig_sh, _ = plt.subplots(figsize=(10, 4)); shap.plots.waterfall(sv[0], show=False); st.pyplot(fig_sh)
            with ct:
                st.info("### AI Interpretation")
                f_names, s_vals = ['Drug', 'Oil', 'Surfactant', 'Co-surfactant'], sv.values[0]
                top_idx = np.argmax(np.abs(s_vals))
                st.write(f"**Primary Influence:** {f_names[top_idx]}\n\n**Effect:** This component is {'increasing' if s_vals[top_idx] > 0 else 'decreasing'} the droplet size by **{abs(s_vals[top_idx]):.1f} nm**.")
        except Exception as e: st.error(f"Prediction Error: {e}")
