              
                                 MORPH-PPMS Framework
                  
                        ███╗   ███╗ ██████╗ ██████╗ ██████╗ ██╗  ██╗
                        ████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ██║
                        ██╔████╔██║██║   ██║██████╔╝██████╔╝███████║
                        ██║╚██╔╝██║██║   ██║██╔══██╗██╔═══╝ ██╔══██║
                        ██║ ╚═╝ ██║╚██████╔╝██║  ██║██║     ██║  ██║
                        ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
                  
                           [AI] --> ( proteome reversal ) --> [ Score ]
                       
                    MORPH: AI-Driven Molecular Morphology & DPI Predictor
                    [■] TASK: Drug-Protein Interaction [■] VER: 1.1.0
                    [■] AUTH: Shixuan.Z & ZhenQiu.L    [■] SYS: Mac M or Win 10

    
# 🧬 MORPH: An AI-Driven Framework for Proteome Reversal Score (PPMS) & Drug-Protein Interaction Prediction

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![R Version](https://img.shields.io/badge/R-4.3%2B-green.svg)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/License-MIT-darkgray.svg)](LICENSE)

**MORPH ** is an advanced, multi-target drug discovery (MTDD) framework designed to decipher complex metabolic comorbidity clusters (MCCs) and prioritize synergistic drug combinations via deep learning and network pharmacology. 

---

## 🚀 Key Features

* **Proteomic-Level Prediction:** Leverages ESM2 protein language models and molecular fingerprints to predict directional drug-protein interaction (DPI) polarities (activation vs. inhibition).
* **Network Propagation Engine:** Implements the **Proteome Perturbation Modulation Score (PPMS)** via Random Walk with Restart (RWR) and PPI interactome dynamics to evaluate macro-level regulatory synergy.
* **Dual-Drive Paradigm:** Harmonizes algorithm-driven virtual screening with a knowledge-driven expert clinical auditing layer to address multi-drug intolerance (MDI) constraints.

---

## 🛠️ Environmental Prerequisites

MORPH requires a hybrid Python and R execution environment. Ensure both runtimes meet the following specifications:

## Python Environment (v3.8+)

Install the required scientific and deep learning dependencies via `pip`:
```bash
pip install torch torch_geometric pandas numpy scikit-learn networkx tqdm matplotlib seaborn lightgbm joblib rdkit
```
## ⚠️💡 Prepare documents

Please download the Rawdata-Tool/ folder: https://zenodo.org/uploads/21309729 

## 📂 Repository & Data Architecture

```text
.
├── MORPH.py                  # DPI execution engine (Deep Learning & Gradient Boosting inference)
├── PPMS.py                   # Network propagation & proteomic synergy scoring framework
├── Rawdata/                  # Static reference repositories (Keep relative structure intact)
│   └── EMS2_MF/              # Embedded checkpoints for GNN and PPI interactome matrices
│   ├── Ref/                  # Standard mapping files for Drugs and Proteins
│   └── trained_models/       # Model pre-training file
│   ├── PPMS/                 # MCC mapping in PPMS analysis
└── User/                     # Workspace designated for target inputs and computational exports
```

## Input Data Formats

MORPH dynamically accommodates two structural input configurations via self-contained string normalization:

SMILES + Protein Identity: The target file must explicitly define a SMILES and GeneSymbol column.

Nomenclature Identifiers: The target file must explicitly define a ChemicalName and GeneSymbol column.


## 💻 Workflow & Execution Guide
### Step 1: Directional Drug-Protein Interaction (DPI) Prediction

Execute the feed-forward deep learning and LightGBM pipeline to determine pair-wise binding polarities:

```bash
python MORPH.py -i User/Text.csv
```
> Output Specification:
The generated file is compiled inside User/Prediction_Costaware_Text.csv containing a standardized four-variable matrix:
Drug: Normalized chemical/small-molecule nomenclature identifier.
Protein: Standardized Human Gene Symbol.
Action: Interaction regulatory polarity (1 denotes targeted activation/upregulation; -1 denotes targeted inhibition/downregulation).
Confidence: Probability distribution profiling of the predicted interaction.


###Step 2: Systemic Hierarchical Prioritization (PPMS Matrix Calculation)

Propagate localized perturbation vectors across the target disease interactome spectrum to yield unified synergy scores:

```bash
python PPMS.py \
  --dti User/Prediction_Costaware_Text.csv \
  --disease User/Cluster1_table2.csv \
  --ppi User/Cluster1_table3.csv \
  --pathway User/Cluster1_table4.csv \
  --cluster MCC1
```
> Output Specification
The network propagation matrix writes the final evaluation file directly into the root workspace:TotalScore_MCC1_Prediction_Costaware_Text.csv (Features fully normalized, cross-cluster comparable metrics including PRS, PRS_syn, and multi-stage penalty variables).


## Acknowledgments

We extend our deepest gratitude to the clinical pharmacists and collaborators who made the stringent double-blind validation possible:

Shixuan.Z  (Algorithm Implementation)
ZhenQiu.L (Algorithm Implementation)
Jingru.G (Clinical Pharmacist, Independent Blinded Rater)
Shun.S (Clinical Pharmacist, Independent Blinded Rater)

## ✉️ Contact & Citatio
Author: Shixuan Zhang
Institution: Fudan University, Shanghai, China
Email: sxzhang21@m.fudan.edu.cn



