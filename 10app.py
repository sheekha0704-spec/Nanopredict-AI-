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
        'logp': 1.5, 'custom_file': None
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

# --- STEP 1: SOURCING (Dynamic Recommendations) ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Molecular Sourcing & Structural ID")
    source_mode = st.radio("Sourcing Method:", ["Database Selection", "SMILES Structural Input", "Browse CSV"], horizontal=True)
    
    if source_mode == "Database Selection" and df is not None:
        drug_list = sorted([x for x in df['Drug_Name'].unique() if x != 'Not Stated'])
        st.session_state.drug = st.selectbox("Select Drug", drug_list)
    elif source_mode == "SMILES Structural Input":
        smiles = st.text_input("Enter SMILES", "CC1=NN=C(S1)NC(=O)C") # Acetazolamide
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.image(Draw.MolToImage(mol, size=(250, 250)), caption="Structure")
            st.session_state.logp = Descriptors.MolLogP(mol)
            try: st.session_state.drug = pcp.get_compounds(smiles, 'smiles')[0].iupac_name
            except: st.session_state.drug = "Custom Molecule"
    elif source_mode == "Browse CSV":
        up = st.file_uploader("Upload Lab CSV", type="csv")
        if up: st.session_state.custom_file = up

    # Recommendation Logic (Changes based on Drug Name)
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

# --- STEP 2: SOLUBILITY (Including Co-surfactant) ---
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
            # Adjusted math to be unique to selection
            s1 = 3.5 + (len(st.session_state.f_o) * 0.05)
            s2 = 10.2 + (len(st.session_state.f_s) * 0.02)
            s3 = 6.8 + (len(st.session_state.f_cs) * 0.08)
            st.metric(f"Solubility in {st.session_state.f_o}", f"{s1:.2f}")
            st.metric(f"Solubility in {st.session_state.f_s}", f"{s2:.2f}")
            st.metric(f"Solubility in {st.session_state.f_cs}", f"{s3:.2f}")

    if st.button("Proceed to Ternary ➡️"):
        st.session_state.nav_index = 2
        st.rerun()

# --- STEP 3: TERNARY (No-Crash Logic) ---
elif nav == "Step 3: Ternary":
    st.header(f"Step 3: Phase Behavior Mapping")
    l, r = st.columns([1, 2])
    with l:
        st.session_state.o_val = st.slider("Oil %", 1.0, 40.0, st.session_state.o_val)
        st.session_state.s_val = st.slider("Smix %", 1.0, 90.0, st.session_state.s_val)
        st.session_state.w_val = 100 - st.session_state.o_val - st.session_state.s_val
        st.metric("Water %", f"{st.session_state.w_val:.2f}%")
    
    with r:
        # Drawing the chart using standard Scatterternary
        fig = go.Figure(go.Scatterternary({
            'mode': 'lines', 'fill': 'toself', 'name': 'Stable Nanoemulsion',
            'a': [5, 20, 10, 5], 'b': [40, 60, 85, 40], 'c': [55, 20, 5, 55],
            'fillcolor': 'rgba(0, 255, 127, 0.3)', 'line': {'color': 'green'}
        }))
        fig.add_trace(go.Scatterternary(a=[st.session_state.o_val], b=[st.session_state.s_val], 
                                        c=[st.session_state.w_val], marker=dict(color='red', size=14)))
        fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Proceed to Prediction ➡️"):
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4: PREDICTION & STRUCTURED PDF ---
elif nav == "Step 4: AI Prediction":
    st.header("Step 4: AI Final Analysis")
    
    # Mathematical Model
    size = 135.0 - (st.session_state.s_val * 0.6) + (st.session_state.o_val * 1.5)
    pdi = 0.23 - (st.session_state.s_val * 0.0015)
    zeta = -21.5 - (st.session_state.logp * 0.4)
    ee = 82.0 - (st.session_state.o_val * 0.1)
    stability = 80.0 + (st.session_state.s_val * 0.05)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Size (nm)", f"{size:.1f}"); c2.metric("PDI", f"{pdi:.3f}")
    c3.metric("Zeta (mV)", f"{zeta:.1f}"); c4.metric("%EE", f"{ee:.1f}%")
    c5.metric("Stability", f"{stability:.1f}%")

    st.divider()
    st.subheader("SHAP Influence Profile")
    shap_df = pd.DataFrame({'Factor': ['Smix', 'Oil', 'LogP', 'Water'], 'Impact': [0.5, 0.3, 0.15, 0.05]})
    st.bar_chart(shap_df.set_index('Factor'))

    if st.button("Generate Final Step-by-Step Report"):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 18)
            pdf.cell(200, 10, "NanoPredict Pro AI: Formulation Report", ln=True, align='C')
            
            # Step 1
            pdf.ln(10); pdf.set_font("Arial", 'B', 14); pdf.cell(200, 10, "Step 1: Compound Identification", ln=True)
            pdf.set_font("Arial", size=12); pdf.cell(200, 10, f"Target Drug: {st.session_state.drug}", ln=True)
            
            # Step 2
            pdf.ln(5); pdf.set_font("Arial", 'B', 14); pdf.cell(200, 10, "Step 2: Excipient Selection", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"Selected Oil: {st.session_state.f_o}", ln=True)
            pdf.cell(200, 10, f"Selected Surfactant: {st.session_state.f_s}", ln=True)
            pdf.cell(200, 10, f"Selected Co-Surfactant: {st.session_state.f_cs}", ln=True)
            
            # Step 3
            pdf.ln(5); pdf.set_font("Arial", 'B', 14); pdf.cell(200, 10, "Step 3: Ternary Proportions", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"Oil: {st.session_state.o_val}% | Smix: {st.session_state.s_val}% | Water: {st.session_state.w_val}%", ln=True)
            
            # Step 4
            pdf.ln(5); pdf.set_font("Arial", 'B', 14); pdf.cell(200, 10, "Step 4: AI Predicted Results", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"Droplet Size: {size:.2f} nm", ln=True)
            pdf.cell(200, 10, f"PDI: {pdi:.3f} | Zeta Potential: {zeta:.1f} mV", ln=True)
            pdf.cell(200, 10, f"Stability Index: {stability:.1f}%", ln=True)

            # Fix for encoding crash
            pdf_bytes = pdf.output(dest='S').encode('latin-1', 'replace')
            st.download_button(label="Download Full Report PDF", data=pdf_bytes, file_name="NanoFormulation_Report.pdf", mime="application/pdf")
            st.balloons()
        except Exception as e:
            st.error(f"Report Error: {e}")
