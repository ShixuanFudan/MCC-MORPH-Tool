import os
import sys
import warnings
import multiprocessing
import time
import pickle
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import re
from typing import Dict, Any, Optional, Tuple, List
import io
import shutil
import torch
from torch.utils.data import Dataset, DataLoader

# Suppress system and third-party environment warnings during execution
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Unable to retrieve source.*")
warnings.filterwarnings("ignore", module="torch._jit_internal")
os.environ['PYTHONWARNINGS'] = 'ignore'


# Resolve resource paths for standalone PyInstaller deployment binaries
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# Disable PyTorch JIT execution to mitigate serialization conflicts with PyInstaller
os.environ['PYTORCH_JIT'] = '0'


def dummy_jit_script(obj, *args, **kwargs): return obj


def dummy_overload(*args, **kwargs): return lambda func: func


torch.jit.script = dummy_jit_script
torch.jit.interface = dummy_jit_script
torch.jit._overload = dummy_overload

# Initialize molecular informatics environment and evaluate RDKit tractability
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import BulkTanimotoSimilarity
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit unavailable. Similarity-based fingerprint matching will fall back to exact string identity.")
    Chem = None
    BulkTanimotoSimilarity = None

# Initialize geometric deep learning modules and fallbacks
try:
    from torch_geometric.nn import SAGEConv, GATConv
    from torch_geometric.data import Data
except ImportError:
    print("Warning: torch_geometric not detected. Utilizing fallback data structures.")


    class Data:
        def __init__(self, x, edge_index):
            self.x = x
            self.edge_index = edge_index

warnings.filterwarnings('ignore')


# =====================================================================
# Model Architecture Definitions
# =====================================================================

class EnhancedTrainingConfig:
    """Configuration hyperparameters for GNN representation learning."""

    def __init__(self):
        self.hidden_dim = 512
        self.dropout_rate = 0.2
        self.use_batch_norm = True
        self.num_gcn_layers = 3
        self.use_attention = True
        self.attention_heads = 8
        self.feature_fusion_method = 'attention'
        self.use_self_supervised = False


class EnhancedGNNFeatureExtractor(torch.nn.Module):
    """Deep learning architecture for high-resolution drug-protein interaction embeddings."""

    def __init__(self, esm2_dim=640, drug_dim=2048, config=None):
        super().__init__()
        self.config = config if config else EnhancedTrainingConfig()
        hidden_dim = self.config.hidden_dim

        self.protein_preprocessor = torch.nn.Sequential(
            torch.nn.Linear(esm2_dim, hidden_dim * 2),
            torch.nn.BatchNorm1d(hidden_dim * 2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(self.config.dropout_rate),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(self.config.dropout_rate)
        )

        self.drug_preprocessor = torch.nn.Sequential(
            torch.nn.Linear(drug_dim, hidden_dim * 2),
            torch.nn.BatchNorm1d(hidden_dim * 2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(self.config.dropout_rate),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(self.config.dropout_rate)
        )

        self.gcn_layers = torch.nn.ModuleList()
        for i in range(self.config.num_gcn_layers):
            if i < self.config.num_gcn_layers - 1 and self.config.use_attention:
                self.gcn_layers.append(
                    GATConv(hidden_dim, hidden_dim // max(1, self.config.attention_heads),
                            heads=self.config.attention_heads, dropout=self.config.dropout_rate)
                )
            else:
                self.gcn_layers.append(SAGEConv(hidden_dim, hidden_dim))

        if self.config.feature_fusion_method == 'attention':
            self.feature_attention = torch.nn.MultiheadAttention(
                hidden_dim, num_heads=4, dropout=self.config.dropout_rate, batch_first=True
            )
            self.feature_fusion = self._attention_fusion
        else:
            self.fusion_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 3, hidden_dim * 2),
                torch.nn.BatchNorm1d(hidden_dim * 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.config.dropout_rate),
                torch.nn.Linear(hidden_dim * 2, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.config.dropout_rate)
            )
            self.feature_fusion = self._nonlinear_fusion

        self.output_dim = hidden_dim

    def _attention_fusion(self, features):
        p_base, p_ppi, d_base = features
        feature_seq = torch.stack([p_base, p_ppi, d_base], dim=1)
        attended_features, _ = self.feature_attention(feature_seq, feature_seq, feature_seq)
        return attended_features.mean(dim=1)

    def _nonlinear_fusion(self, features):
        p_base, p_ppi, d_base = features
        return self.fusion_mlp(torch.cat([p_base, p_ppi, d_base], dim=1))

    def forward(self, protein_features, drug_features, ppi_data, protein_indices, drug_indices, return_features=True):
        if not hasattr(ppi_data, 'protein_ppi_all') or ppi_data.protein_ppi_all is None:
            protein_base_all = self.protein_preprocessor(protein_features)
            protein_ppi_all = protein_base_all

            if hasattr(ppi_data, 'edge_index') and ppi_data.edge_index.shape[1] > 0:
                for gcn_layer in self.gcn_layers:
                    protein_ppi_all = gcn_layer(protein_ppi_all, ppi_data.edge_index)
                    if isinstance(gcn_layer, GATConv) and protein_ppi_all.dim() == 3:
                        protein_ppi_all = protein_ppi_all.mean(dim=1)
                    protein_ppi_all = torch.nn.functional.leaky_relu(protein_ppi_all, 0.1)

            ppi_data.protein_base_all = protein_base_all
            ppi_data.protein_ppi_all = protein_ppi_all
        else:
            protein_base_all = ppi_data.protein_base_all
            protein_ppi_all = ppi_data.protein_ppi_all

        drug_base_all = self.drug_preprocessor(drug_features)
        batch_protein_base = protein_base_all[protein_indices]
        batch_protein_ppi = protein_ppi_all[protein_indices]
        batch_drug_base = drug_base_all[drug_indices]

        if return_features:
            return self.feature_fusion((batch_protein_base, batch_protein_ppi, batch_drug_base))
        return batch_protein_base, batch_protein_ppi, batch_drug_base


# =====================================================================
# Computational Utilities & Preprocessing
# =====================================================================

def normalize_id_for_matching(id_value):
    """Normalizes identifiers to achieve case-insensitive, alphanumeric token consistency."""
    if pd.isna(id_value): return None
    id_str = str(id_value).strip().lower()
    id_str = re.sub(r'[^\w\s]', ' ', id_str)
    id_str = re.sub(r'\s+', ' ', id_str).strip()
    if '.' in id_str:
        try:
            id_str = str(int(float(id_str)))
        except ValueError:
            pass
    return id_str


def load_trained_model(model_path, device):
    """Instantiates and loads pretrained GNN model weights."""
    checkpoint = torch.load(model_path, map_location='cpu')
    config = EnhancedTrainingConfig()
    if 'config' in checkpoint:
        for key, value in checkpoint['config'].items():
            if hasattr(config, key): setattr(config, key, value)
    model = EnhancedGNNFeatureExtractor(esm2_dim=640, drug_dim=2048, config=config)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model = model.to(device).eval()
    return model, config


def load_lightgbm_model_dict(model_path):
    """Extracts LightGBM downstream classifiers and embedded standardization objects."""
    if not os.path.exists(model_path): return None, None, None
    try:
        model_dict = joblib.load(model_path)
        if 'model' in model_dict:
            return model_dict['model'], {k: model_dict.get(k) for k in
                                         ['scaler', 'quantile_transformer', 'pca']}, model_dict.get('feature_dim', 512)
        return None, None, None
    except Exception:
        return None, None, None


def preprocess_features(features, preprocessors, feature_dim):
    """Transforms raw latent tensors via deterministic pre-processing pipelines."""
    processed_features = features.copy()
    if processed_features.shape[1] != feature_dim:
        if processed_features.shape[1] > feature_dim:
            processed_features = processed_features[:, :feature_dim]
        elif processed_features.shape[1] < feature_dim:
            padding = np.zeros((processed_features.shape[0], feature_dim - processed_features.shape[1]))
            processed_features = np.hstack([processed_features, padding])

    if preprocessors.get('pca'): processed_features = preprocessors['pca'].transform(processed_features)
    if preprocessors.get('quantile_transformer'): processed_features = preprocessors['quantile_transformer'].transform(
        processed_features)
    if preprocessors.get('scaler'): processed_features = preprocessors['scaler'].transform(processed_features)
    return processed_features


def load_and_preprocess_new_data(esm2_path, drug_fp_path, y_path=None, ppi_path=None, chunk_size=50000):
    """Ingests high-dimensional omics descriptors and initializes matching matrices."""
    protein_esm2_features = {}
    for chunk in pd.read_csv(esm2_path, chunksize=chunk_size, encoding='utf-8', on_bad_lines='skip',
                             encoding_errors="replace"):
        chunk['GeneSymbol_original'] = chunk['GeneSymbol'].astype(str)
        chunk['GeneSymbol'] = chunk['GeneSymbol_original'].apply(normalize_id_for_matching)
        chunk = chunk.dropna(subset=['GeneSymbol'])
        embedding_cols = [col for col in chunk.columns if col.startswith('Embedding_')]
        for prot_norm, features in zip(chunk['GeneSymbol'], chunk[embedding_cols].values):
            protein_esm2_features[prot_norm] = features.astype(np.float32)

    drug_fp_features = {}
    for chunk in pd.read_csv(drug_fp_path, chunksize=chunk_size, encoding='utf-8', on_bad_lines='skip',
                             encoding_errors="replace"):
        chunk['ChemicalName_original'] = chunk['ChemicalName'].astype(str)
        chunk['ChemicalName'] = chunk['ChemicalName_original'].apply(normalize_id_for_matching)
        chunk = chunk.dropna(subset=['ChemicalName'])
        fp_cols = [col for col in chunk.columns if col.startswith('FP_')]
        for drug_norm, features in zip(chunk['ChemicalName'], chunk[fp_cols].values):
            drug_fp_features[drug_norm] = features.astype(np.float32)

    labels, original_pairs, y_df = {}, {}, None
    if y_path and os.path.exists(y_path):
        with open(y_path, 'r', encoding='utf-8') as f:
            temp_content = f.read()
        temp_cols = pd.read_csv(io.StringIO(temp_content), nrows=1).columns.tolist()
        use_cols = ['ChemicalName', 'GeneSymbol'] + [c for c in ['Relation', 'SMILES', 'Sequence'] if c in temp_cols]

        y_df = pd.read_csv(io.StringIO(temp_content), usecols=use_cols, encoding_errors="replace")
        y_df['ChemicalName_original'] = y_df['ChemicalName'].astype(str).str.strip()
        y_df['GeneSymbol_original'] = y_df['GeneSymbol'].astype(str).str.strip()
        y_df['ChemicalName_norm'] = y_df['ChemicalName_original'].apply(normalize_id_for_matching)
        y_df['GeneSymbol_norm'] = y_df['GeneSymbol_original'].apply(normalize_id_for_matching)
        y_df = y_df.dropna(subset=['ChemicalName_norm', 'GeneSymbol_norm'])

        for _, row in y_df.iterrows():
            dn, pn = row['ChemicalName_norm'], row['GeneSymbol_norm']
            if dn in drug_fp_features and pn in protein_esm2_features:
                labels[(dn, pn)] = row.get('Relation', 0) if 'Relation' in y_df.columns else 0
                original_pairs[(dn, pn)] = (row['ChemicalName_original'], row['GeneSymbol_original'])

    ppi_df = None
    if ppi_path and os.path.exists(ppi_path):
        ppi_df = pd.read_csv(ppi_path)
        ppi_df['protein1_Gene'] = ppi_df['protein1_Gene'].astype(str).apply(normalize_id_for_matching)
        ppi_df['protein2_Gene'] = ppi_df['protein2_Gene'].astype(str).apply(normalize_id_for_matching)
        ppi_df = ppi_df.dropna(subset=['protein1_Gene', 'protein2_Gene'])

    return protein_esm2_features, drug_fp_features, labels, ppi_df, y_df, original_pairs


def build_extended_graph_data(original_graph_data, new_protein_features, new_drug_features, new_ppi_df, new_labels,
                              device):
    """Maps dynamic interactome extensions onto static structural graph bases."""
    extended_graph_data = original_graph_data.copy()
    original_protein_to_idx = extended_graph_data.get('protein_to_idx', {})
    original_drug_to_idx = extended_graph_data.get('drug_to_idx', {})
    original_all_proteins = extended_graph_data.get('all_proteins', [])
    original_all_drugs = extended_graph_data.get('all_drugs', [])

    new_proteins = [p for p in new_protein_features if p not in original_protein_to_idx]
    new_drugs = [d for d in new_drug_features if d not in original_drug_to_idx]

    if not new_proteins and not new_drugs: return extended_graph_data

    all_proteins_extended = original_all_proteins + new_proteins
    protein_to_idx_extended = {p: i for i, p in enumerate(all_proteins_extended)}
    all_drugs_extended = original_all_drugs + new_drugs
    drug_to_idx_extended = {d: i for i, d in enumerate(all_drugs_extended)}

    protein_matrices = [
        extended_graph_data['protein_esm2_matrix']] if 'protein_esm2_matrix' in extended_graph_data else []
    protein_matrices.extend([torch.FloatTensor(new_protein_features[p]).unsqueeze(0) for p in new_proteins])
    protein_esm2_matrix_extended = torch.cat(protein_matrices, dim=0)

    drug_matrices = [extended_graph_data['drug_fp_matrix']] if 'drug_fp_matrix' in extended_graph_data else []
    drug_matrices.extend([torch.FloatTensor(new_drug_features[d]).unsqueeze(0) for d in new_drugs])
    drug_fp_matrix_extended = torch.cat(drug_matrices, dim=0)

    ppi_edges = []
    if 'ppi_data' in extended_graph_data and hasattr(extended_graph_data['ppi_data'], 'edge_index'):
        for src, dst in extended_graph_data['ppi_data'].edge_index.t().tolist():
            if src < len(original_all_proteins) and dst < len(original_all_proteins):
                p1, p2 = original_all_proteins[src], original_all_proteins[dst]
                if p1 in protein_to_idx_extended and p2 in protein_to_idx_extended:
                    ppi_edges.append([protein_to_idx_extended[p1], protein_to_idx_extended[p2]])

    if new_ppi_df is not None:
        for _, row in new_ppi_df.iterrows():
            p1, p2 = str(row['protein1_Gene']), str(row['protein2_Gene'])
            if p1 in protein_to_idx_extended and p2 in protein_to_idx_extended:
                s, d = protein_to_idx_extended[p1], protein_to_idx_extended[p2]
                ppi_edges.extend([[s, d], [d, s]])

    ppi_edge_index = torch.tensor(list(set(tuple(e) for e in ppi_edges)),
                                  dtype=torch.long).t().contiguous() if ppi_edges else torch.empty((2, 0),
                                                                                                   dtype=torch.long)
    ppi_data_extended = Data(x=torch.zeros((len(all_proteins_extended), protein_esm2_matrix_extended.size(1))),
                             edge_index=ppi_edge_index)

    extended_graph_data.update({
        'protein_esm2_matrix': protein_esm2_matrix_extended,
        'drug_fp_matrix': drug_fp_matrix_extended,
        'protein_to_idx': protein_to_idx_extended,
        'drug_to_idx': drug_to_idx_extended,
        'all_proteins': all_proteins_extended,
        'all_drugs': all_drugs_extended,
        'ppi_data': ppi_data_extended,
        'labels': new_labels
    })
    return extended_graph_data


class PredictionDataset(Dataset):
    """Dataset encapsulation for streaming pair-wise computational screening tasks."""

    def __init__(self, labels, protein_to_idx, drug_to_idx, original_pairs=None):
        self.labels = labels
        self.protein_to_idx = protein_to_idx
        self.drug_to_idx = drug_to_idx
        self.original_pairs = original_pairs or {}
        self.valid_pairs, self.original_valid_pairs, self.labels_list = [], [], []

        for (dn, pn), label in labels.items():
            if pn in protein_to_idx and dn in drug_to_idx:
                self.valid_pairs.append((dn, pn))
                self.original_valid_pairs.append(self.original_pairs.get((dn, pn), (dn, pn)))
                self.labels_list.append(label)

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        dn, pn = self.valid_pairs[idx]
        return self.protein_to_idx[pn], self.drug_to_idx[dn], self.labels_list[idx], idx


def extract_gnn_features(model, graph_data, dataset, device, batch_size=128):
    """Executes feed-forward generation of cross-attention deep proteomic embeddings."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features, all_norm_pairs, all_original_pairs = [], [], []

    with torch.no_grad():
        for p_idx, d_idx, _, b_idx in tqdm(dataloader, desc="Extract graph features"):
            features = model(
                graph_data['protein_esm2_matrix'].to(device),
                graph_data['drug_fp_matrix'].to(device),
                graph_data['ppi_data'],
                p_idx.to(device),
                d_idx.to(device),
                return_features=True
            )
            all_features.append(features.cpu().numpy())
            for idx in b_idx:
                all_norm_pairs.append(dataset.valid_pairs[idx])
                all_original_pairs.append(dataset.original_valid_pairs[idx])

    return np.vstack(all_features) if all_features else np.array([]), all_norm_pairs, all_original_pairs


def run_single_prediction_task(task_name, lgbm_model_path, features_array, norm_pairs, original_pairs, y_df,
                               base_output_dir):
    """Executes gradient boosted decision tree screening on computed graph tensor vectors."""
    lightgbm_model, preprocessors, feature_dim = load_lightgbm_model_dict(lgbm_model_path)
    if not lightgbm_model: return None

    processed_features = preprocess_features(features_array, preprocessors, feature_dim)

    try:
        if hasattr(lightgbm_model, 'predict_proba'):
            predictions_proba = lightgbm_model.predict_proba(processed_features)
            predictions = lightgbm_model.predict(processed_features)
        else:
            preds = lightgbm_model.predict(processed_features)
            if preds.ndim == 1:
                predictions_proba = np.vstack([1 - preds, preds]).T
                predictions = (preds > 0.5).astype(int)
            else:
                predictions_proba = preds
                predictions = np.argmax(preds, axis=1)
    except Exception:
        return None

    num_classes = predictions_proba.shape[1]
    max_probs = np.max(predictions_proba, axis=1) if num_classes > 1 else predictions_proba[:, 0]

    results = []
    for i, (_, (do, po)) in enumerate(zip(norm_pairs, original_pairs)):
        results.append({
            'ChemicalName': str(do).strip(),
            'GeneSymbol': str(po).strip(),
            f'{task_name}_prediction': predictions[i],
            f'{task_name}_probability': float(max_probs[i])
        })

    results_df = pd.DataFrame(results)

    if y_df is not None and 'ChemicalName_original' in y_df.columns:
        y_copy = y_df.copy()
        y_copy['ChemicalName'] = y_copy['ChemicalName_original'].astype(str).str.strip()
        y_copy['GeneSymbol'] = y_copy['GeneSymbol_original'].astype(str).str.strip()
        return pd.merge(y_copy, results_df, on=['ChemicalName', 'GeneSymbol'], how='left')
    return results_df


def calculate_smiles_similarity_match(input_smiles, drug_map):
    """Computes Bulk-Tanimoto structural similarities using Morgan chemical fingerprints."""
    if not Chem or not BulkTanimotoSimilarity: return None, 0.0, None
    input_mol = Chem.MolFromSmiles(input_smiles)
    if not input_mol: return None, 0.0, None
    input_fp = GetMorganFingerprint(input_mol, 2)

    mols = [Chem.MolFromSmiles(s) for s in drug_map['SMILES']]
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    target_fps = [GetMorganFingerprint(mols[i], 2) for i in valid_indices]

    similarities = BulkTanimotoSimilarity(input_fp, target_fps)
    if not similarities: return None, 0.0, None
    max_sim = max(similarities)
    best_id = drug_map['ChemicalName_map'].iloc[valid_indices[similarities.index(max_sim)]]

    return (str(best_id) if max_sim >= 0.9 else None), max_sim, str(best_id)


def create_prediction_input_from_smi_seq_file(input_file_path, drug_map_file, protein_map_file):
    """Generates standard screening matrices from raw SMILES and sequence definitions."""
    input_df = pd.read_csv(input_file_path)
    drug_map = pd.read_csv(drug_map_file).dropna(subset=['SMILES', 'ChemicalName']).rename(
        columns={'ChemicalName': 'ChemicalName_map'})
    protein_map = pd.read_csv(protein_map_file).dropna(subset=['Sequence', 'GeneSymbol'])
    protein_id_to_seq = dict(zip(protein_map['GeneSymbol'].astype(str).str.strip(), protein_map['Sequence']))

    matched_data = []
    for _, row in input_df.iterrows():
        drug_id, _, _ = calculate_smiles_similarity_match(str(row['SMILES']).strip(), drug_map)
        seq = protein_id_to_seq.get(str(row['GeneSymbol']).strip())
        if drug_id and seq:
            matched_data.append({
                'ChemicalName': drug_id, 'GeneSymbol': row['GeneSymbol'],
                'SMILES': row['SMILES'], 'Sequence': seq
            })

    if not matched_data: return None, ''
    temp_path = os.path.join('User/', os.path.basename(input_file_path).replace('.csv', '_ID_Matched.csv'))
    pd.DataFrame(matched_data).to_csv(temp_path, index=False)
    return temp_path, os.path.basename(input_file_path)


# =====================================================================
# Main Pipeline Core Execution
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="MORPH Framework: AI-driven Virtual Screening Pipeline")
    parser.add_argument('-i', type=str, required=True, help='Target CSV file for predictive validation')
    parser.add_argument('--drug_map_file', type=str, default=resource_path('Rawdata/Ref/Drug.csv'))
    parser.add_argument('--protein_map_file', type=str, default=resource_path('Rawdata/Ref/Prot.csv'))
    args = parser.parse_args()

    logo = r"""
    --------------------------------------------------------------------
                             MORPH Framework
    
          ███╗   ███╗ ██████╗ ██████╗ ██████╗ ██╗  ██╗
          ████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ██║
          ██╔████╔██║██║   ██║██████╔╝██████╔╝███████║
          ██║╚██╔╝██║██║   ██║██╔══██╗██╔═══╝ ██╔══██║
          ██║ ╚═╝ ██║╚██████╔╝██║  ██║██║     ██║  ██║
          ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
    
               [AI] --> ( proteome reversal ) --> [ Score ]
         
          MORPH: AI-Driven Molecular Morphology & protein expression Predictor
          [■] TASK: Drug-Protein Interaction [■] VER: 1.1.0
          [■] AUTH: Shixuan.Z & ZhenQiu.L    [■] SYS: Mac M or Win 10
    --------------------------------------------------------------------"""
    print("\n")
    for line in logo.split('\n'):
        sys.stdout.write(f"\033[94m{line}\033[0m\n")
        sys.stdout.flush()
        time.sleep(0.01)

    gnn_model_path = resource_path('Rawdata/enhanced_features/enhanced_feature_extractor_model.pth')
    graph_data_path = resource_path('Rawdata/enhanced_features/graph_data.pkl')
    original_lgbm_model_path = resource_path('Rawdata/trained_models/best_model.pkl')
    esm2_path = resource_path('Rawdata/EMS2_MF/protein_esm2_features_final_new.csv')
    drug_fp_path = resource_path('Rawdata/EMS2_MF/drug_fp_features_final_new.csv')
    ppi_path = resource_path('Rawdata/enhanced_features/PPI.csv')

    os.makedirs('User/', exist_ok=True)

    with open(args.i, 'r', encoding='utf-8') as f:
        cols = pd.read_csv(io.StringIO(f.read()), nrows=1).columns.tolist()
    if 'ChemicalName' in cols and 'GeneSymbol' in cols:
        y_path_final, y_df_name = args.i, os.path.basename(args.i)
    elif 'SMILES' in cols and 'GeneSymbol' in cols:
        y_path_final, y_df_name = create_prediction_input_from_smi_seq_file(args.i, args.drug_map_file,
                                                                            args.protein_map_file)
        if not y_path_final: return
    else:
        print("Error: Required input headers ('ChemicalName'/'GeneSymbol') are missing.")
        return

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    gnn_model, _ = load_trained_model(gnn_model_path, device)

    with open(graph_data_path, 'rb') as f:
        original_graph_data = pickle.load(f)

    new_prot, new_drug, labels, ppi, y_df, orig_pairs = load_and_preprocess_new_data(esm2_path, drug_fp_path,
                                                                                     y_path_final, ppi_path)
    if not labels: return

    extended_graph = build_extended_graph_data(original_graph_data, new_prot, new_drug, ppi, labels, device)
    dataset = PredictionDataset(labels, extended_graph['protein_to_idx'], extended_graph['drug_to_idx'], orig_pairs)

    with torch.no_grad():
        p_base = gnn_model.protein_preprocessor(extended_graph['protein_esm2_matrix'].to(device))
        p_ppi = p_base
        if hasattr(extended_graph['ppi_data'], 'edge_index') and extended_graph['ppi_data'].edge_index.shape[1] > 0:
            for l in gnn_model.gcn_layers:
                p_ppi = torch.nn.functional.leaky_relu(l(p_ppi, extended_graph['ppi_data'].edge_index.to(device)), 0.1)
        extended_graph['ppi_data'].protein_base_all = p_base
        extended_graph['ppi_data'].protein_ppi_all = p_ppi

    features, norm_pairs, orig_pairs_list = extract_gnn_features(gnn_model, extended_graph, dataset, device)

    # Execute downstream primary prediction model and parse target variables
    if os.path.exists(original_lgbm_model_path):
        merged_final = run_single_prediction_task('Original', original_lgbm_model_path, features, norm_pairs,
                                                  orig_pairs_list, y_df, 'User/')

        if merged_final is not None:
            # 1. Map columns onto requested standardized nomenclature variables
            column_mapping = {
                'ChemicalName': 'Drug',
                'GeneSymbol': 'Protein',
                'Original_prediction': 'Action',
                'Original_probability': 'Confidence'
            }
            merged_final = merged_final.rename(columns=column_mapping)

            # Keep only the explicitly requested target four columns
            target_columns = ['Drug', 'Protein', 'Action', 'Confidence']
            merged_final = merged_final[[col for col in target_columns if col in merged_final.columns]]

            # 2. Concurrently cast binary zero predictions into negative-one labels
            if 'Action' in merged_final.columns:
                merged_final['Action'] = merged_final['Action'].replace(0, -1)

            out_path = os.path.join('User/', f'Prediction_Costaware_{y_df_name}')
            merged_final.to_csv(out_path, index=False)
            print(f"\nPrediction completed. Target metrics exported successfully to: {out_path}")
        else:
            print("Error: Downstream prediction execution failed.")
    else:
        print("Error: Target checkpoint directory is inaccessible.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()