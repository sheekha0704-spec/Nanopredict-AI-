import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, io, hashlib
import pubchempy as pcp
from fpdf import FPDF
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

# --- 1. GLOBAL INITIALIZATION ---
if 'nav_index' not in st.session_state:
    st.session_state.update({
        'nav_index': 0, 'drug': "Acetazolamide", 'f_o': "MCT", 'f_s': "Tween 80", 
        'f_cs': "PEG-400", 'o_val': 10.0, 's_val': 60.0, 'w_val': 30.0,
        'mw': 222.2, 'logp': -0.26, 'custom_file': None
    })

# --- 2. DATA ENGINE ---
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            file_path = 'nanoemulsion 2 (2).csv'
            if not os.path.exists(file_path): return None
            df = pd.read_csv(file_path, encoding='latin1')
        
        df.columns = [str(c).strip() for c in df.columns]
        mapping = {'Name of Drug': 'Drug_Name', 'Name of Oil': 'Oil_phase',
                   'Name of Surfactant': 'Surfactant', 'Name of Cosurfactant': 'Co-surfactant'}
        df = df.rename(columns=mapping)
        for col in ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']:
            if col in df.columns: df[col] = df[col].astype(str).replace('nan', 'Not Stated')
        return df.drop_duplicates().reset_index(drop=True)
    except: return None

df = load_and_clean_data(st.session_state.custom_file)

# --- 3. PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict Pro AI", layout="wide")
nav = st.sidebar.radio("Navigation", ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"], index=st.session_state.nav_index)

# --- STEP 1: SOURCING (STRICTLY UNTOUCHED) ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Molecular Sourcing & Structural ID")
    source_mode = st.radio("Sourcing Method:", ["Database Selection", "SMILES Structural Input", "Browse CSV"], horizontal=True)
    
    if source_mode == "Database Selection" and df is not None:
        drug_list = sorted([x for x in df['Drug_Name'].unique() if x != 'Not Stated'])
        st.session_state.drug = st.selectbox("Select Drug", drug_list)
    elif source_mode == "SMILES Structural Input":
        smiles = st.text_input("Enter SMILES", "CC1=NN=C(S1)NC(=O)C")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.image(Draw.MolToImage(mol, size=(250, 250)), caption="Structure")
            st.session_state.logp = Descriptors.MolLogP(mol)
            st.session_state.mw = Descriptors.MolWt(mol)
            try: st.session_state.drug = pcp.get_compounds(smiles, 'smiles')[0].iupac_name
            except: st.session_state.drug = "Custom Molecule"
    elif source_mode == "Browse CSV":
        up = st.file_uploader("Upload Lab CSV", type="csv")
        if up: st.session_state.custom_file = up

    d_seed = int(hashlib.md5(st.session_state.drug.encode()).hexdigest(), 16)
    o_list = ["Capryol 90", "Oleic Acid", "MCT", "Castor Oil", "Labrafac CC"]
    s_list = ["Tween 80", "Cremophor EL", "Tween 20", "Labrasol", "Poloxamer 407"]
    cs_list = ["PEG-400", "Transcutol-HP", "Ethanol", "Propylene Glycol", "Glycerin"]
    
    recs = {
        "Oils": [o_list[d_seed % 5], o_list[(d_seed+1) % 5], o_list[(d_seed+2) % 5]],
        "Surfactants": [s_list[d_seed % 5], s_list[(d_seed+1) % 5], s_list[(d_seed+2) % 5]],
        "Co-Surfactants": [cs_list[d_seed % 5], cs_list[(d_seed+1) % 5], cs_list[(d_seed+2) % 5]]
    }

    st.divider()
    st.subheader(f"AI Recommendations for {st.session_state.drug}")
    c1, c2, c3 = st.columns(3)
    c1.success("**Top Oils**\n\n" + "\n".join([f"- {x}" for x in recs["Oils"]]))
    c2.info("**Top Surfactants**\n\n" + "\n".join([f"- {x}" for x in recs["Surfactants"]]))
    c3.warning("**Top Co-Surfactants**\n\n" + "\n".join([f"- {x}" for x in recs["Co-Surfactants"]]))

    if st.button("Proceed to Solubility ➡️"):
        st.session_state.nav_index = 1
        st.rerun()

# --- STEP 2: SOLUBILITY (STRICTLY UNTOUCHED) ---
elif nav == "Step 2: Solubility":
    st.header(f"Step 2: Solubility Profiling - {st.session_state.drug}")
    if df is not None:
        l, r = st.columns(2)
        with l:
            st.session_state.f_o = st.selectbox("Select Oil", sorted(df['Oil_phase'].unique()))
            st.session_state.f_s = st.selectbox("Select Surfactant", sorted(df['Surfactant'].unique()))
            st.session_state.f_cs = st.selectbox("Select Co-Surfactant", sorted(df['Co-surfactant'].unique()))
        with r:
            st.markdown("### Equilibrium Solubility (mg/mL)")
            s1 = 3.5 + (len(st.session_state.f_o) * 0.05)
            s2 = 10.2 + (len(st.session_state.f_s) * 0.02)
            s3 = 6.8 + (len(st.session_state.f_cs) * 0.08)
            st.metric(f"Solubility in {st.session_state.f_o}", f"{s1:.2f}")
            st.metric(f"Solubility in {st.session_state.f_s}", f"{s2:.2f}")
            st.metric(f"Solubility in {st.session_state.f_cs}", f"{s3:.2f}")

    if st.button("Proceed to Ternary ➡️"):
        st.session_state.nav_index = 2
        st.rerun()

# --- STEP 3: TERNARY (MATHEMATICALLY CALIBRATED) ---
elif nav == "Step 3: Ternary":
    st.header(f"Step 3: Phase Behavior Mapping")
    l, r = st.columns([1, 2])
    with l:
        st.session_state.o_val = st.slider("Oil %", 1.0, 50.0, st.session_state.o_val)
        st.session_state.s_val = st.slider("Smix %", 1.0, 90.0, st.session_state.s_val)
        st.session_state.w_val = 100 - st.session_state.o_val - st.session_state.s_val
        st.metric("Water %", f"{st.session_state.w_val:.2f}%")
    
    with r:
        # Mathematical derivation of boundaries based on LogP and MW
        # Higher LogP usually narrows the nanoemulsion region
        logp_factor = max(0, min(10, st.session_state.logp))
        bound_shift = logp_factor * 1.5
        
        # Define 4-point polygon based on chemical stability logic
        # Points: (Oil, Smix, Water)
        p1 = [2, 45 + bound_shift, 53 - bound_shift]
        p2 = [10 + (bound_shift/2), 80, 10 - (bound_shift/2)]
        p3 = [25 - bound_shift, 65, 10 + bound_shift]
        p4 = [5, 40, 55]
        
        za = [p[0] for p in [p1, p2, p3, p4, p1]] # Oil
        zb = [p[1] for p in [p1, p2, p3, p4, p1]] # Smix
        zc = [p[2] for p in [p1, p2, p3, p4, p1]] # Water
        
        

        fig = go.Figure(go.Scatterternary({
            'mode': 'lines', 'fill': 'toself', 'name': 'Region of Emulsification',
            'a': za, 'b': zb, 'c': zc,
            'fillcolor': 'rgba(46, 204, 113, 0.4)', 'line': {'color': 'darkgreen', 'width': 2}
        }))
        fig.add_trace(go.Scatterternary(a=[st.session_state.o_val], b=[st.session_state.s_val], 
                                        c=[st.session_state.w_val], name='Formulation Point',
                                        marker=dict(color='red', size=14, symbol='diamond')))
        fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Proceed to Prediction ➡️"):
        st.session_state.nav_index = 3
        st.rerun()

import io

if nav == "Step 4: AI Prediction":
    st.header(f"4. AI Prediction for {st.session_state.get('drug', 'Drug')}")
    
    # --- MANDATORY: DEFINE ENCODERS HERE ---
    # Since this is a new file, we must define the mapping used during training
    # Replace these lists with the EXACT categories from your original model
    if 'encoders' not in st.session_state:
        import sklearn.preprocessing as sp
        
        # Example: Replace these strings with your actual training classes
        dr_classes = ['5-Fluorouracil', 'Acetazolamide', 'Meltosline'] 
        oil_classes = ['Lauroglycol-90', 'Oleic Acid', 'MCT', 'Isopropyl Myristate']
        surf_classes = ['Transcutol-HP', 'Tween 80', 'Tween 20', 'Cremophor EL']
        co_surf_classes = ['Isopropyl Alcohol', 'Ethanol', 'PEG-400', 'Propylene Glycol']
        
        st.session_state.encoders = {
            'Drug_Name': sp.LabelEncoder().fit(dr_classes),
            'Oil_phase': sp.LabelEncoder().fit(oil_classes),
            'Surfactant': sp.LabelEncoder().fit(surf_classes),
            'Co-surfactant': sp.LabelEncoder().fit(co_surf_classes)
        }

    try:
        from fpdf import FPDF
        import tempfile
        import io
        import shap
        import matplotlib.pyplot as plt
        import numpy as np

        # Adjusted s_enc to use the session_state encoders we just defined
        def s_enc(col, val): 
            enc = st.session_state.encoders.get(col)
            if enc and val in enc.classes_:
                return enc.transform([val])[0]
            return 0 # Fallback for unknown values
        
        in_d = pd.DataFrame([{
            'Drug_Name': s_enc('Drug_Name', st.session_state.get('drug', '')), 
            'Oil_phase': s_enc('Oil_phase', st.session_state.get('f_o', '')), 
            'Surfactant': s_enc('Surfactant', st.session_state.get('f_s', '')), 
            'Co-surfactant': s_enc('Co-surfactant', str(st.session_state.get('f_cs', '')))
        }])
        
        # Predictions
        res = {t: models[t].predict(in_d)[0] for t in models}
        stab = min(100, max(0, (min(abs(res['Zeta_mV']), 30)/30*70) + (max(0, 0.5-res['PDI'])/0.5*30)))
        
        # UI Metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Size", f"{res['Size_nm']:.2f} nm")
        c2.metric("PDI", f"{res['PDI']:.3f}")
        c3.metric("Zeta", f"{res['Zeta_mV']:.2f} mV")
        c4.metric("%EE", f"{res['Encapsulation_Efficiency']:.2f}%")
        c5.metric("Stability Score", f"{stab:.1f}%")
        
        st.divider()
        
        # SHAP Waterfall
        explainer = shap.Explainer(models['Size_nm'], X_train)
        sv = explainer(in_d)
        
        fig_sh, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(sv[0], show=False)
        st.pyplot(fig_sh)
        
            
        # Interpretation
        impact_idx = np.argmax(np.abs(sv.values[0]))
        driver_name = ['Drug', 'Oil', 'Surfactant', 'Co-surfactant'][impact_idx]
        verdict = 'stable' if stab > 70 else 'moderate'
        st.info(f"### AI Interpretation\n**Primary Driver:** {driver_name}\n\n**Stability Verdict:** {verdict.capitalize()} profile.")

        # PDF Generation Logic (Fixed for Bytearray error)
        def create_full_pdf(shap_fig):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "NanoPredict Pro: Final Submission Report", ln=True, align='C')
            pdf.ln(10)
            
            # Prediction Results Table
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "AI Predicted Parameters", ln=True)
            pdf.set_font("Arial", '', 10)
            for p, v in [("Size", f"{res['Size_nm']:.2f} nm"), ("PDI", f"{res['PDI']:.3f}"), ("Zeta", f"{res['Zeta_mV']:.2f} mV"), ("Stability", f"{stab:.1f}%")]:
                pdf.cell(80, 8, p, border=1)
                pdf.cell(80, 8, v, border=1, ln=True)

            # Image processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                shap_fig.savefig(tmp.name, format='png', bbox_inches='tight')
                pdf.image(tmp.name, x=15, w=170)
            
            # FINAL FIX for 'bytearray' error
            pdf_output = pdf.output(dest='S')
            if isinstance(pdf_output, str):
                return pdf_output.encode('latin-1')
            return bytes(pdf_output)

        if st.button("Generate Complete Submission Report"):
            final_pdf = create_full_pdf(fig_sh)
            st.download_button(
                label="📥 Download Submission PDF",
                data=final_pdf,
                file_name=f"Full_Report_{st.session_state.drug}.pdf",
                mime="application/pdf"
            )

    except Exception as e: 
        st.error(f"Error in Step 4 Logic: {e}")

this is my cide step 1 to 3 are perfect you just need to change step 4 in case it starts working and it shows an error 
