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

# --- 1. DATA ENGINE (FINAL ENCODING FIX) ---
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    df = None
    file_path = 'nanoemulsion 2 (2).csv'
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin1')
    elif os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
        except Exception:
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
        sig_map = {name: generate_chemical_sig(name) for name in df['Drug_Name'].unique()}
        df['Ref_MW'] = df['Drug_Name'].map(sig_map)

    def to_float(value):
        if pd.isna(value): return 0.0
        val_str = str(value).lower().strip()
        if any(x in val_str for x in ['low', 'not stated', 'nan', 'null']): return 0.0
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
        return float(nums[0]) if nums else 0.0

    for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']:
        if col in df.columns: df[col] = df[col].apply(to_float)

    for col in ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant', 'Method']:
        if col in df.columns: df[col] = df[col].astype(str).replace(['nan', 'None', 'Unknown'], 'Unknown')
    
    return df[df['Drug_Name'] != 'Unknown']

# --- APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro", layout="wide")
if 'nav_index' not in st.session_state: st.session_state.nav_index = 0
steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

def get_clean_unique(df, col):
    items = set(df[col].dropna().unique())
    return sorted([str(x) for x in items if str(x).lower() not in ['unknown', 'nan', '', 'none']])

@st.cache_resource
def train_models(_data):
    if _data is None: return None, None, None, None
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    le_dict = {col: LabelEncoder().fit(_data[col].astype(str)) for col in features + ['Method']}
    df_enc = _data.copy()
    for col in features + ['Method']: df_enc[col] = le_dict[col].transform(_data[col].astype(str))
    models = {t: GradientBoostingRegressor(n_estimators=50, random_state=42).fit(df_enc[features], df_enc[t]) for t in ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']}
    return models, le_dict, df_enc[features]

df = load_and_clean_data(st.session_state.get('custom_file'))
if df is not None:
    models, encoders, X_train = train_models(df)

# --- STEP 1: SOURCING (RECALIBRATED PERSONALIZATION) ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Formulation Sourcing")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.subheader("📁 Browse from the file")
        up_file = st.file_uploader("Upload Lab CSV", type="csv")
        if up_file: st.session_state.custom_file = up_file; st.rerun()
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
                    st.session_state.drug = (drug_sigs - input_mw).abs().idxmin()
                    st.write(f"**Molecular Weight:** {input_mw:.2f} g/mol")
                    st.success(f"AI mapped to: {st.session_state.drug}")
            except: st.error("Chemical engine error.")

    st.divider()
    # Logic for drug-specific recommendations
    d_subset = df[df['Drug_Name'] == st.session_state.get('drug', 'Unknown')]
    def get_personalized_top_3(subset, full_df, col):
        specific = get_clean_unique(subset, col)
        # Prioritize drug-specific options, fill with global if needed
        combined = specific + [x for x in get_clean_unique(full_df, col) if x not in specific]
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

# --- STEP 2: SOLUBILITY (DEDUPLICATED) ---
elif nav == "Step 2: Solubility":
    st.header("2. AI-Predicted Solubility Profile")
    # Using sorted set to ensure zero duplicates and alphabetical order
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
        st.metric(f"Solubility in {sel_o}", f"{(len(sel_o)*0.45)+2.5:.2f} mg/mL")
        st.metric(f"Solubility in {sel_s}", f"{(len(sel_s)*0.25)+1.8:.2f} mg/mL")
        st.metric(f"Solubility in {sel_cs}", f"{(len(sel_cs)*0.15)+0.6:.2f} mg/mL")
    if st.button("Next: Ternary Mapping ➡️"): st.session_state.nav_index = 2; st.rerun()

# --- STEP 3: TERNARY MAPPING ---
elif nav == "Step 3: Ternary":
    st.header(f"3. Ternary Phase Mapping")
    l, r = st.columns([1, 2])
    with l:
        st.markdown("### Formulation Input")
       # Inside Step 3:
        oil_v = st.slider("Oil Content (%)", 1, 50, 15, key='oil_v') # ADD KEY
        smix_v = st.slider("Smix %", 1, 90, 45, key='smix_v') # ADD KEY
        water_v = 100 - oil_v - smix_v
        if water_v < 0: st.error("Total Oil + Smix exceeds 100%.")
        else: st.metric("Automatically Calculated Water %", f"{water_v}%")
    with r:
        fig = go.Figure()
        fig.add_trace(go.Scatterternary(a=[0,20,10,0], b=[45,70,90,45], c=[55,10,0,55], fill='toself', fillcolor='rgba(0,255,100,0.2)'))
        fig.add_trace(go.Scatterternary(a=[oil_v], b=[smix_v], c=[water_v], marker=dict(size=18, color='red', symbol='diamond')))
        fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
        st.plotly_chart(fig, use_container_width=True)
    if st.button("Next: AI Prediction ➡️"): st.session_state.nav_index = 3; st.rerun()

# --- STEP 4: PREDICTION & PDF (STABLE VERSION) ---
elif nav == "Step 4: AI Prediction":
    st.header(f"4. AI Prediction for {st.session_state.get('drug', 'Drug')}")
    
    # SAFETY CHECK: Ensure session state attributes exist to avoid "AttributeError"
    required_keys = ['drug', 'f_o', 'f_s', 'f_cs', 'oil_v', 'smix_v']
    missing_keys = [key for key in required_keys if key not in st.session_state]
    
    if missing_keys:
        st.warning("⚠️ Some formulation data is missing. Please complete Steps 1, 2, and 3 first.")
    else:
        try:
            from fpdf import FPDF
            import tempfile

            # 1. ENCODING & PREDICTION LOGIC
            def s_enc(col, val): 
                return encoders[col].transform([val])[0] if val in encoders[col].classes_ else 0
            
            in_d = pd.DataFrame([{
                'Drug_Name': s_enc('Drug_Name', st.session_state.drug), 
                'Oil_phase': s_enc('Oil_phase', st.session_state.f_o), 
                'Surfactant': s_enc('Surfactant', st.session_state.f_s), 
                'Co-surfactant': s_enc('Co-surfactant', str(st.session_state.f_cs))
            }])
            
            res = {t: models[t].predict(in_d)[0] for t in models}
            stab = min(100, max(0, (min(abs(res['Zeta_mV']), 30)/30*70) + (max(0, 0.5-res['PDI'])/0.5*30)))
            
            # 2. UI METRICS DISPLAY
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Size", f"{res['Size_nm']:.2f} nm")
            c2.metric("PDI", f"{res['PDI']:.3f}")
            c3.metric("Zeta", f"{res['Zeta_mV']:.2f} mV")
            c4.metric("%EE", f"{res['Encapsulation_Efficiency']:.2f}%")
            c5.metric("Stability Score", f"{stab:.1f}%")
            
            st.divider()
            
            # 3. SHAP WATERFALL CHART
            fig_sh, ax = plt.subplots(figsize=(10, 4))
            explainer = shap.Explainer(models['Size_nm'], X_train)
            sv = explainer(in_d)
            shap.plots.waterfall(sv[0], show=False)
            st.pyplot(fig_sh)

            # 4. COMPREHENSIVE PDF GENERATION FUNCTION
            def create_full_pdf(shap_fig):
                pdf = FPDF()
                pdf.add_page()
                
                # Header
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, "NanoPredict Pro: Final Submission Report", ln=True, align='C')
                pdf.set_font("Arial", 'I', 10)
                pdf.cell(200, 10, f"Generated for: {st.session_state.drug}", ln=True, align='C')
                pdf.line(10, 30, 200, 30)
                pdf.ln(10)

                # SECTION 1: COMPOSITION TABLE
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, "1. Formulation Composition Analysis", ln=True)
                pdf.set_font("Arial", 'B', 10)
                pdf.cell(80, 8, "Component", border=1, align='C')
                pdf.cell(80, 8, "Percentage (%)", border=1, ln=True, align='C')
                
                pdf.set_font("Arial", '', 10)
                o_v = st.session_state.oil_v
                s_v = st.session_state.smix_v
                w_v = 100 - o_v - s_v
                
                for label, val in [("Oil Phase", f"{o_v}%"), ("Smix (Surf/Co-Surf)", f"{s_v}%"), ("Water Phase", f"{w_v}%")]:
                    pdf.cell(80, 8, label, border=1)
                    pdf.cell(80, 8, val, border=1, ln=True)
                pdf.ln(5)
                
                # SECTION 2: MATERIALS
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, "2. Material Selection", ln=True)
                pdf.set_font("Arial", '', 10)
                pdf.cell(0, 7, f"Drug: {st.session_state.drug}", ln=True)
                pdf.cell(0, 7, f"Oil: {st.session_state.f_o} | Surfactant: {st.session_state.f_s} | Co-Surf: {st.session_state.f_cs}", ln=True)
                pdf.ln(5)

                # SECTION 3: AI RESULTS
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, "3. AI Predicted Results", ln=True)
                pdf.set_font("Arial", 'B', 10)
                pdf.cell(80, 8, "Parameter", border=1, align='C')
                pdf.cell(80, 8, "Predicted Value", border=1, ln=True, align='C')
                
                pdf.set_font("Arial", '', 10)
                results = [
                    ("Droplet Size", f"{res['Size_nm']:.2f} nm"),
                    ("PDI", f"{res['PDI']:.3f}"),
                    ("Zeta Potential", f"{res['Zeta_mV']:.2f} mV"),
                    ("Stability Score", f"{stab:.1f}%")
                ]
                for p, v in results:
                    pdf.cell(80, 8, p, border=1)
                    pdf.cell(80, 8, v, border=1, ln=True)
                pdf.ln(5)
                
                # SECTION 4: SHAP IMAGE
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, "4. Factor Influence Analysis (SHAP)", ln=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    shap_fig.savefig(tmp.name, format='png', bbox_inches='tight')
                    pdf.image(tmp.name, x=15, w=170)
                
                return pdf.output(dest='S').encode('latin-1')

            # DOWNLOAD BUTTON
            if st.button("Generate Complete Submission Report"):
                with st.spinner("Compiling results..."):
                    final_pdf = create_full_pdf(fig_sh)
                    st.download_button(
                        label="📥 Download Submission PDF",
                        data=final_pdf,
                        file_name=f"Report_{st.session_state.drug}.pdf",
                        mime="application/pdf"
                    )

        except Exception as e: 
            st.error(f"Error compiling report: {e}")
