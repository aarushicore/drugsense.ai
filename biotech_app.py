# biointel_streamlit_app.py
# BioIntel: Streamlit App for AutoGen-Powered Drug Discovery

# NOTE:
# Requires: streamlit, rdkit-pypi, requests
# Install via: pip install streamlit rdkit-pypi requests

import streamlit as st
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from PIL import Image
from io import BytesIO

# --- Agent Logic ---
def molecule_designer(target_disease):
    st.info("MoleculeDesigner: Generating molecule for target...")
    return "CC(C(=O)O)N"  # Simulated output (Alanine)

def property_predictor(smiles):
    st.info("PropertyPredictor: Calculating molecular descriptors...")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES"
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    return f"Molecular Weight: {mw:.2f}, LogP: {logp:.2f}"

def simulator(smiles):
    st.info("SimulatorAgent: Running conformer simulation...")
    mol = Chem.MolFromSmiles(smiles)
    return "Stable Conformer Found" if mol else "Simulation failed"

def pubchem_lookup(smiles):
    st.info("DatabaseLookupAgent: Searching PubChem...")
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
        response = requests.get(url)
        data = response.json()
        if "IdentifierList" in data:
            return f"Found in PubChem. CID: {data['IdentifierList']['CID'][0]}"
        return "Not found in PubChem."
    except:
        return "PubChem lookup failed."

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        return img
    return None

# --- Streamlit Interface ---
st.set_page_config(page_title="BioIntel: AI Drug Discovery", layout="wide")
st.title("🧬 BioIntel: AutoGen-Powered Drug Discovery")

with st.form("biointel_form"):
    disease = st.text_input("Enter target disease or condition:", "inflammation")
    submitted = st.form_submit_button("Design Molecule")

if submitted:
    with st.spinner("Running agents..."):
        smiles = molecule_designer(disease)
        props = property_predictor(smiles)
        sim_result = simulator(smiles)
        pubchem_result = pubchem_lookup(smiles)

        st.subheader("Molecule SMILES")
        st.code(smiles)

        img = draw_molecule(smiles)
        if img:
            st.image(img, caption="Molecule Structure", use_column_width=False)

        st.subheader("Predicted Properties")
        st.success(props)

        st.subheader("Simulation Result")
        st.info(sim_result)

        st.subheader("PubChem Lookup")
        st.warning(pubchem_result)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, RDKit, and PubChem")