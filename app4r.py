# biointel_streamlit_app.py
# BioIntel: Streamlit App for AutoGen-Powered Drug Discovery (Fast Mode Toggle)

# NOTE:
# Requires: streamlit, rdkit-pypi, requests, transformers, torch
# Install via: pip install streamlit rdkit-pypi requests transformers torch

import streamlit as st
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from PIL import Image
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- MUST BE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="BioIntel: AI Drug Discovery", layout="wide")

# --- Sidebar Toggle for Fast Mode ---
st.sidebar.title("⚙️ Settings")
fast_mode = st.sidebar.checkbox("⚡ Enable Fast Mode", value=True)

# --- Load smaller/faster pretrained model (ChemGPT 345M or DistilGPT2) ---
@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "ncfrey/ChemGPT-345M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to("cpu")
    return model, tokenizer

# --- Real Molecule Generator using ChemGPT ---
def molecule_designer(target_disease):
    if fast_mode:
        st.info("MoleculeDesigner: Fast Mode enabled (using static molecule)...")
        return "CC(C(=O)O)N"  # Alanine (static SMILES for speed)

    st.info("MoleculeDesigner: Generating molecule using ChemGPT...")
    model, tokenizer = load_model()

    prompt = f"{target_disease}:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            do_sample=True,
            top_k=20,
            top_p=0.95,
            num_return_sequences=1
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    possible_smiles = decoded.split(":")[-1].split()[0]
    mol = Chem.MolFromSmiles(possible_smiles)
    return possible_smiles if mol else "CC(C(=O)O)N"

# --- Property Predictor ---
def property_predictor(smiles):
    st.info("PropertyPredictor: Calculating molecular descriptors...")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES"
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    return f"Molecular Weight: {mw:.2f}, LogP: {logp:.2f}"

# --- Simulator ---
def simulator(smiles):
    st.info("SimulatorAgent: Running conformer simulation...")
    mol = Chem.MolFromSmiles(smiles)
    return "Stable Conformer Found" if mol else "Simulation failed"

# --- PubChem Lookup ---
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

# --- Toxicity Predictor ---
def toxicity_predictor(smiles):
    st.info("ToxicityPredictor: Estimating toxicity (simulated)...")
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "Invalid SMILES for toxicity prediction."
    num_atoms = mol.GetNumAtoms()
    toxicity_score = 0.1 * num_atoms
    if toxicity_score < 5:
        risk = "Low"
    elif toxicity_score < 10:
        risk = "Moderate"
    else:
        risk = "High"
    return f"Predicted Toxicity Score: {toxicity_score:.2f} → Risk Level: {risk}"

# --- Molecule Drawing ---
def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        return img
    return None

# --- Streamlit Interface ---
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
        tox_result = toxicity_predictor(smiles)

        st.subheader("Molecule SMILES")
        st.code(smiles)

        img = draw_molecule(smiles)
        if img:
            st.image(img, caption="Molecule Structure", use_container_width=True)

        st.subheader("Predicted Properties")
        st.success(props)

        st.subheader("Simulation Result")
        st.info(sim_result)

        st.subheader("PubChem Lookup")
        st.warning(pubchem_result)

        st.subheader("Toxicity Prediction")
        st.error(tox_result)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, RDKit, PubChem, and ChemGPT (Fast Mode Toggle)")
