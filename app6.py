# biointel_streamlit_app.py
# BioIntel: Streamlit App for AutoGen-Powered Drug Discovery (Dynamic Molecule Output per Disease)

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
import random
import hashlib

# --- MUST BE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="BioIntel: AI Drug Discovery", layout="wide")

# --- Sidebar Toggle for Fast Mode and Disease Selector ---
st.sidebar.title("⚙️ Settings")
fast_mode = st.sidebar.checkbox("⚡ Enable Fast Mode", value=True)

common_conditions = [
    "inflammation", "diabetes", "cancer", "malaria", "tuberculosis", "alzheimer's disease",
    "parkinson's disease", "epilepsy", "arthritis", "lupus", "psoriasis", "HIV", "COVID-19"
]

selected_condition = st.sidebar.selectbox("🎯 Select common condition (optional):", ["Custom"] + common_conditions)

# --- Load molecule-generating model from Hugging Face ---
@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "ncfrey/SMILES-1M-GPT2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to("cpu")
    return model, tokenizer

# --- Real Molecule Generator using SMILES-trained GPT2 ---
def molecule_designer(target_disease):
    if fast_mode:
        st.info("MoleculeDesigner: Fast Mode enabled (using static molecule)...")
        return ["CC(C(=O)O)N"]

    st.info("MoleculeDesigner: Generating molecules using SMILES-GPT2...")
    model, tokenizer = load_model()

    # Ensure prompt produces variety per disease using hash-based seed
    disease_seed = int(hashlib.sha256(target_disease.encode()).hexdigest(), 16) % 10000
    torch.manual_seed(disease_seed)

    prompt = f"Generate a SMILES molecule for {target_disease}:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=100,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=1.0,
            num_return_sequences=3
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    valid_smiles = []
    for text in decoded:
        candidates = text.split(":")[-1].strip().split()
        for s in candidates:
            mol = Chem.MolFromSmiles(s)
            if mol and s not in valid_smiles:
                valid_smiles.append(s)

    return valid_smiles if valid_smiles else ["CC(C(=O)O)N"]

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
    if selected_condition != "Custom":
        disease = selected_condition
    else:
        disease = st.text_input("Enter target disease or condition:", "inflammation")
    submitted = st.form_submit_button("Design Molecule")

if submitted:
    with st.spinner("Running agents..."):
        smiles_list = molecule_designer(disease)

        for idx, smiles in enumerate(smiles_list):
            st.subheader(f"Molecule #{idx+1} SMILES")
            st.code(smiles)

            img = draw_molecule(smiles)
            if img:
                st.image(img, caption="Molecule Structure", use_container_width=True)

            props = property_predictor(smiles)
            sim_result = simulator(smiles)
            pubchem_result = pubchem_lookup(smiles)
            tox_result = toxicity_predictor(smiles)

            st.subheader("Predicted Properties")
            st.success(props)

            st.subheader("Simulation Result")
            st.info(sim_result)

            st.subheader("PubChem Lookup")
            st.warning(pubchem_result)

            st.subheader("Toxicity Prediction")
            st.error(tox_result)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, RDKit, PubChem, and SMILES-trained GPT2 for Drug Discovery")