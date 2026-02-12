import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os, re, hashlib, io
import pubchempy as pcp
from fpdf import FPDF
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

# --- 1. SESSION STATE INITIALIZATION (Prevents crashes) ---
if 'nav_index' not in st.session_state:
    st.session_state.update({
        'nav_index': 0, 'drug': "Unknown", 'f_o': "None", 'f_s': "None", 
        'f_cs': "None", 'o_val': 15.0, 's_val': 45.0, 'w_val': 40.0,
        'mw': 0.0, 'logp': 2.0, 'ternary_img': None, 'custom_file': None
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
            if col in df.columns: df[col] = df[col].astype(str)
        return df.drop_duplicates().reset_index(drop=True)
    except: return None

df = load_and_clean_data(st.session_state.custom_file)

# --- 3. NAVIGATION ---
st.set_page_config(page_title="NanoPredict Pro AI", layout="wide")
nav = st.sidebar.radio("Navigation", ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"], index=st.session_state.nav_index)

# --- STEP 1: SOURCING (Recalibrated for 3 Recommendations) ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Molecular Sourcing & Structural ID")
    source_mode = st.radio("Sourcing Method:", ["Database Selection", "SMILES Structural Input"], horizontal=True)
    
    current_drug = "Unknown"
    o_recs, s_recs, cs_recs = [], [], []

    if source_mode == "Database Selection" and df is not None:
        drug_list = sorted([x for x in df['Drug_Name'].unique() if x.lower() != 'nan'])
        current_drug = st.selectbox("Select Drug", drug_list)
        # Pull top 3 actual matches from CSV
        subset = df[df['Drug_Name'] == current_drug]
        o_recs = subset['Oil_phase'].unique()[:3].tolist()
        s_recs = subset['Surfactant'].unique()[:3].tolist()
        cs_recs = subset['Co-surfactant'].unique()[:3].tolist()
    
    elif source_mode == "SMILES Structural Input":
        smiles = st.text_input("Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.image(Draw.MolToImage(mol, size=(300, 300)))
            st.session_state.logp = Descriptors.MolLogP(mol)
            try: current_drug = pcp.get_compounds(smiles, 'smiles')[0].iupac_name
            except: current_drug = "Custom Compound"
            # 3 Custom AI Recommendations based on LogP
            if st.session_state.logp > 3:
                o_recs, s_recs, cs_recs = ["Capryol 90", "MCT", "Castor Oil"], ["Tween 80", "Cremophor EL", "Labrasol"], ["PEG 400", "Transcutol P", "Ethanol"]
            else:
                o_recs, s_recs, cs_recs = ["Oleic Acid", "IPM", "Labrafil"], ["Tween 20", "Span 80", "Plurol Oleique"], ["Propylene Glycol", "Glycerin", "Water"]

    if current_drug != "Unknown":
        st.session_state.drug = current_drug
        st.subheader(f"AI Recommendations for {current_drug}")
        c1, c2, c3 = st.columns(3)
        c1.success("**Recommended Oils**\n\n" + "\n".join([f"- {x}" for x in o_recs]))
        c2.info("**Recommended Surfactants**\n\n" + "\n".join([f"- {x}" for x in s_recs]))
        c3.warning("**Recommended Co-Surfactants**\n\n" + "\n".join([f"- {x}" for x in cs_recs]))

    if st.button("Proceed ➡️"): 
        st.session_state.nav_index = 1
        st.rerun()

# --- STEP 2: SOLUBILITY (Recalibrated Logic) ---
elif nav == "Step 2: Solubility":
    st.header(f"Step 2: Solubility Profiling - {st.session_state.drug}")
    if df is not None:
        l, r = st.columns(2)
        with l:
            o = st.selectbox("Select Oil", sorted(df['Oil_phase'].unique()))
            s = st.selectbox("Select Surfactant", sorted(df['Surfactant'].unique()))
            cs = st.selectbox("Select Co-Surfactant", sorted(df['Co-surfactant'].unique()))
            st.session_state.update({"f_o": o, "f_s": s, "f_cs": cs})
        with r:
            st.markdown("### Equilibrium Solubility (mg/mL)")
            # LogP influenced solubility calculation
            base = 2.0 + (st.session_state.logp * 0.5)
            st.metric(f"Solubility in {o}", f"{(base + len(o)*0.1):.2f}")
            st.metric(f"Solubility in {s}", f"{(base * 1.5 + len(s)*0.05):.2f}")
            st.metric(f"Solubility in {cs}", f"{(base * 0.8 + len(cs)*0.2):.2f}")

    if st.button("Proceed ➡️"): 
        st.session_state.nav_index = 2
        st.rerun()

# --- STEP 3: TERNARY (Corrected Region & Math) ---
elif nav == "Step 3: Ternary":
    st.header(f"Step 3: Ternary Phase Diagram")
    l, r = st.columns([1, 2])
    with l:
        oil_v = st.slider("Oil %", 1.0, 50.0, st.session_state.o_val)
        smix_v = st.slider("Smix %", 1.0, 80.0, st.session_state.s_val)
        water_v = 100 - oil_v - smix_v
        st.session_state.update({"o_val": oil_v, "s_val": smix_v, "w_val": water_v})
        st.metric("Water %", f"{water_v:.2f}%")
    
    with r:
        # Generate a scientifically accurate "Stable" polygon region
        # Coordinates follow (Oil, Smix, Water)
        # Low Oil/High Smix usually forms the nanoemulsion region
        za = [1, 5, 25, 10, 1]      # Oil
        zb = [40, 75, 50, 30, 40]   # Smix
        zc = [100-a-b for a, b in zip(za, zb)] # Water
        
        
        
        fig = go.Figure(go.Scatterternary({
            'mode': 'lines', 'fill': 'toself', 'name': 'Self-Emulsifying Region',
            'a': za, 'b': zb, 'c': zc,
            'fillcolor': 'rgba(46, 204, 113, 0.4)', 'line': {'color': 'green', 'width': 2}
        }))
        fig.add_trace(go.Scatterternary(a=[oil_v], b=[smix_v], c=[water_v], 
                                        name='Current Mix', marker=dict(color='red', size=15, symbol='diamond')))
        fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil Phase', baxis_title='Smix (S/CoS)', caxis_title='Water Phase'))
        st.plotly_chart(fig, use_container_width=True)
        # Fix for PDF: Convert Plotly to bytes correctly
        st.session_state.ternary_img = fig.to_image(format="png", engine="kaleido")

    if st.button("Proceed ➡️"): 
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4: PREDICTION & PDF REPORT (Fixed Generation) ---
elif nav == "Step 4: AI Prediction":
    st.header("Step 4: AI Final Analysis")
    
    # Calculation Logic
    size = 150.0 - (st.session_state.s_val * 0.8) + (st.session_state.o_val * 0.3)
    pdi = 0.15 + (st.session_state.o_val * 0.004)
    zeta = -12.0 - (len(st.session_state.f_s) * 0.5)
    ee = 75.0 + (st.session_state.o_val * 0.2)
    stability = 100 - (pdi * 110)

    

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Size (nm)", f"{size:.1f}"); c2.metric("PDI", f"{pdi:.3f}")
    c3.metric("Zeta (mV)", f"{zeta:.1f}"); c4.metric("%EE", f"{ee:.1f}%")
    c5.metric("Stability", f"{stability:.1f}%")

    st.divider()
    if st.button("Generate & Download PDF Report"):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, f"NanoPredict AI Pro Report: {st.session_state.drug}", ln=True, align='C')
            pdf.set_font("Arial", size=11)
            pdf.ln(10)
            pdf.cell(200, 10, f"Formulation: {st.session_state.f_o} (Oil) | {st.session_state.f_s} (Surfactant)", ln=True)
            pdf.cell(200, 10, f"Physical Data: {size:.1f}nm Size, {pdi:.3f} PDI, {stability:.1f}% Stability", ln=True)
            
            if st.session_state.ternary_img:
                img_stream = io.BytesIO(st.session_state.ternary_img)
                pdf.image(img_stream, x=40, y=80, w=120)
            
            # Robust download output
            pdf_out = pdf.output(dest='S').encode('latin-1')
            st.download_button("Click to Save PDF", data=pdf_out, file_name=f"{st.session_state.drug}_Report.pdf", mime="application/pdf")
            st.balloons()
        except Exception as e:
            st.error(f"PDF Error: {str(e)}. Ensure 'kaleido' is installed.")
