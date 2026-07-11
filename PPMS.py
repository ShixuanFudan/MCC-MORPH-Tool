import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
from collections import defaultdict
from scipy.sparse import diags, eye
from scipy.sparse.linalg import inv
import argparse
import sys
import os
import time

# =================================================================
# 🚀 Algorithmic Hyperparameters Configuration
# =================================================================
RWR_ALPHA = 0.7  # Restart probability for Random Walk with Restart (RWR)
W_COMMON_ACT = 0.7  # Penalty weight for redundant multi-agent activation (C2)
W_BAD_REV = 0.7  # Penalty weight for off-target pathogenic reversal/enhancement (C3)
W_OPP = 0.7  # Penalty weight for antagonistic/conflicting multi-agent perturbations (C4)
W_WRONG_DIR = 0.5  # Penalty weight for disease-exacerbating directional target mismatch (C5)
W_PATHWAY_COVERAGE = 0.7  # Linear mapping weight for downstream pathway coverage (C6)
GAIN_THRESHOLD = 0.05  # Minimum threshold to quantify statistically significant therapeutic gain

# Power scaling coefficient for target annotation confidence profiling
CONFIDENCE_POWER = 3

# Threshold parameter for Phase-I virtual screening to filter top monotherapies
TOP_SINGLE_N = 50


def main():
    parser = argparse.ArgumentParser(
        description='MORPH PPMS Evaluation Engine: Cross-Cluster Normalized Score Aggregation Pipeline')
    parser.add_argument('--dti', type=str, required=True, help='Path to Drug-Target Interaction (DTI) network data')
    parser.add_argument('--disease', type=str, required=True,
                        help='Path to disease-associated protein profiles and pathological roles')
    parser.add_argument('--ppi', type=str, required=True,
                        help='Path to Protein-Protein Interaction (PPI) interactome network')
    parser.add_argument('--pathway', type=str, required=True, help='Path to Protein-Pathway mapping data layers')

    parser.add_argument('--percent', type=str, default='./Rawdata/PPMS/Persent.csv',
                        help='Path to cluster-specific phenotype disease composition matrix')
    parser.add_argument('--atc', type=str, default='./Rawdata/PPMS/ATC_Annote.csv',
                        help='Path to Anatomical Therapeutic Chemical (ATC) clinical indication annotations')
    parser.add_argument('--cluster', type=str, required=True, help='Target Metabolic Comorbidity Cluster (e.g., MCC1)')

    parser.add_argument('--output', type=str, default=None, help='Output path for structural screening statistics')
    args = parser.parse_args()

    # =================================================================
    # 🚀 Professional ASCII Logo Terminal Interface
    # =================================================================
    logo = r"""
--------------------------------------------------------------------
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
--------------------------------------------------------------------"""
    print("\n")
    for line in logo.split('\n'):
        sys.stdout.write(f"\033[93m{line}\033[0m\n")
        sys.stdout.flush()
        time.sleep(0.01)
    print("  ✨ [System Status] Initializing MORPH Network Propagation Engine...")
    print("  ==================================================================\n")

    if args.output is None:
        dti_filename = os.path.basename(args.dti)
        dti_name_only = os.path.splitext(dti_filename)[0]
        args.output = f"TotalScore_{args.cluster}_{dti_name_only}.csv"

    # =================================================================
    # 0. Parsers for Clinical Manifestations & ATC Indications
    # =================================================================
    try:
        print(f"[INFO] Ingesting multi-dimensional clinical manifestation layers...")
        df_percent = pd.read_csv(args.percent)
        df_atc = pd.read_csv(args.atc)

        df_percent['Dies'] = df_percent['Dies'].astype(str).str.strip()
        df_percent['Cluster'] = df_percent['Cluster'].astype(str).str.strip()
        df_percent['Percent_Num'] = df_percent['Percent'].astype(str).str.replace('%', '', regex=False).astype(float)

        # Restrict the primary phenotype set A to indications with prevalence >= 50% within the target cluster
        target_diseases = set(
            df_percent[(df_percent['Cluster'] == args.cluster) & (df_percent['Percent_Num'] >= 50)]['Dies'])

        if not target_diseases:
            print(
                f"[WARNING] No primary disease indications matched the threshold (>= 50%) for Cluster: {args.cluster}")
        else:
            print(f"[SUCCESS] Cluster {args.cluster} Primary Disease Set A (Prevalence >= 50%): {target_diseases}")

        drug_to_diseases = defaultdict(set)
        for _, row in df_atc.iterrows():
            d_name = str(row['Drug names']).strip().lower()
            disease = str(row['Diseases']).strip()
            drug_to_diseases[d_name].add(disease)

    except FileNotFoundError as e:
        print(f"[ERROR] File execution aborted. Resource missing: {e.filename}")
        sys.exit(1)

    # =================================================================
    # 1. Ingestion of Core Interactome & Biological Networks
    # =================================================================
    try:
        print(f"[INFO] Initializing interactome matrices. Target destination: {args.output}")
        df_dti = pd.read_csv(args.dti)
        df_disease = pd.read_csv(args.disease)
        df_ppi = pd.read_csv(args.ppi)
        df_pathway = pd.read_csv(args.pathway)
    except FileNotFoundError as e:
        print(f"[ERROR] Interactome file not found: {e.filename}")
        sys.exit(1)

    prot_to_pathways = defaultdict(list)
    pathway_total_counts = df_pathway.groupby('Pathway')['Protein'].nunique().to_dict()
    all_target_pathways = sorted(list(pathway_total_counts.keys()))
    for _, row in df_pathway.iterrows():
        prot_to_pathways[row['Protein']].append(row['Pathway'])

    # =================================================================
    # 2. Network Topology Optimization & RWR Matrix Precomputations
    # =================================================================
    all_proteins = sorted(list(set(df_ppi['Protein_A']) | set(df_ppi['Protein_B']) |
                               set(df_disease['Protein']) | set(df_dti['Protein'])))
    prot_to_idx = {p: i for i, p in enumerate(all_proteins)}

    G = nx.Graph()
    for _, row in df_ppi.iterrows():
        G.add_edge(row['Protein_A'], row['Protein_B'], weight=row.get('Score', 1.0))
    G.add_nodes_from(all_proteins)

    # Compute sym-normalized adjacency Laplacian operator for structural RWR propagation
    A = nx.adjacency_matrix(G, nodelist=all_proteins)
    d = np.array(A.sum(axis=1)).flatten()
    D_inv = diags(np.where(d > 0, 1.0 / np.sqrt(d), 0))
    W = D_inv @ A @ D_inv
    I = eye(len(all_proteins))

    M_inv = inv(I - (1 - RWR_ALPHA) * W)
    if hasattr(M_inv, "toarray"):
        M = RWR_ALPHA * M_inv.toarray()
    else:
        M = RWR_ALPHA * np.array(M_inv)

    protein_role = dict(zip(df_disease['Protein'], df_disease['Role']))

    # Size normalization parameters to adjust across varying cluster dimensions
    valid_disease_prots = [p for p, role in protein_role.items() if role != 0]
    disease_protein_count = len(valid_disease_prots) if len(valid_disease_prots) > 0 else 1

    total_pw_needed = len(all_target_pathways) if len(all_target_pathways) > 0 else 1

    drug_targets = defaultdict(dict)
    for _, row in df_dti.iterrows():
        boosted_val = row['Action'] * (row['Confidence'] ** CONFIDENCE_POWER)
        drug_targets[row['Drug']][row['Protein']] = boosted_val

    # =================================================================
    # 3. Core Scoring Engine & Network Synergy Calculations
    # =================================================================
    def calculate_score(combo, single_c1_lookup=None, single_prs_lookup=None):
        is_single = len(combo) == 1

        # Calculate clinical ATC phenotype indication coverage ratio
        combo_diseases = set()
        for drug in combo:
            combo_diseases.update(drug_to_diseases.get(drug.lower(), set()))

        covered_set = target_diseases.intersection(combo_diseases)
        coverage_ratio = len(covered_set) / len(target_diseases) if len(target_diseases) > 0 else 0.0
        coverage_str = f"{coverage_ratio:.2%} ({','.join(sorted(list(covered_set))) if covered_set else 'None'})"

        # Construct localized initial perturbation arrays
        p0_matrix = np.zeros((len(combo), len(all_proteins)))
        for i, drug in enumerate(combo):
            for p, val in drug_targets[drug].items():
                p0_matrix[i, prot_to_idx[p]] = val

        # Execute steady-state interactome network propagation matrix multiplication
        prop_effects = p0_matrix @ M
        if prop_effects.ndim == 1:
            prop_effects = prop_effects.reshape(len(combo), -1)
        elif prop_effects.ndim == 0:
            prop_effects = prop_effects.reshape(1, 1)

        total_reversal_gain = 0
        common_act_penalty = 0
        bad_rev_penalty = 0
        opp_penalty = 0
        wrong_dir_penalty = 0

        hit_pathways = set()
        list_success, list_common_act, list_bad_rev, list_conflict, list_wrong_dir = [], [], [], [], []

        for j, prot in enumerate(all_proteins):
            effs = prop_effects[:, j]
            net_eff = np.sum(effs)
            if abs(net_eff) < 1e-4: continue

            role = protein_role.get(prot, 0)
            gain = -(net_eff * role)

            # C1: Target therapeutic reversal profile evaluation
            if gain > GAIN_THRESHOLD:
                list_success.append(f"{prot}(+{round(gain, 2)})")
                total_reversal_gain += gain
                if prot in prot_to_pathways:
                    for pw in prot_to_pathways[prot]:
                        hit_pathways.add(pw)

            # C5: Directional target mismatch penalty (Disease exacerbation phenotype)
            elif gain < -GAIN_THRESHOLD:
                p_val = abs(gain) * W_WRONG_DIR
                wrong_dir_penalty += p_val
                list_wrong_dir.append(f"{prot}(-{round(p_val, 2)})")

            # Combinatorial side-effect and network topology penalty evaluations
            if not is_single:
                act_count = np.sum(effs > 1e-3)
                inh_count = np.sum(effs < -1e-3)

                # C2: Over-activation topology constraints
                if act_count > 1:
                    p_val = (act_count - 1) * W_COMMON_ACT
                    common_act_penalty += p_val
                    list_common_act.append(f"{prot}(-{round(p_val, 2)})")

                # C3: Pathogenic synergy alignment validation
                if (role > 0 and act_count > 1) or (role < 0 and inh_count > 1):
                    p_val = (max(act_count, inh_count) - 1) * W_BAD_REV
                    bad_rev_penalty += p_val
                    list_bad_rev.append(f"{prot}(-{round(p_val, 2)})")

                # C4: Conflicting bi-directional perturbation anomalies
                if act_count > 0 and inh_count > 0:
                    p_val = W_OPP
                    opp_penalty += p_val
                    list_conflict.append(f"{prot}(-{round(p_val, 2)})")

        # C6: Downstream functional pathway coverage score calculation
        hit_pw_count = len(hit_pathways)
        coverage_score_c6 = (hit_pw_count / total_pw_needed) * W_PATHWAY_COVERAGE
        c6_coverage_detail = f"Hit:{hit_pw_count}/{total_pw_needed} ({','.join(sorted(list(hit_pathways))) if hit_pathways else 'None'})"

        # Apply target-level dimensional normalization across cluster variations
        norm_c1_gain = total_reversal_gain / disease_protein_count
        norm_common_act = common_act_penalty / disease_protein_count
        norm_bad_rev = bad_rev_penalty / disease_protein_count
        norm_opp = opp_penalty / disease_protein_count
        norm_wrong_dir = wrong_dir_penalty / disease_protein_count

        # Compute the global aggregate Proteome Perturbation Modulation Score (PPMS/PRS)
        PRS = (norm_c1_gain + coverage_score_c6) - \
              (norm_common_act + norm_bad_rev + norm_opp + norm_wrong_dir)

        # Dual-track synergy matrix metrics initialization
        sc1_syn = 0
        prs_syn = 0

        if not is_single:
            if single_c1_lookup:
                sum_single_c1 = sum(single_c1_lookup.get(d, 0) for d in combo)
                sc1_syn = norm_c1_gain - sum_single_c1
            if single_prs_lookup:
                max_single_prs = max(single_prs_lookup.get(d, 0) for d in combo)
                prs_syn = PRS - max_single_prs

        return {
            'Drugs': ' + '.join(combo),
            'Disease_Coverage': coverage_str,
            'Coverage_Ratio': round(coverage_ratio, 4),
            'PRS': round(PRS, 4),
            'PRS_syn': round(prs_syn, 4),
            'S_C1_Score': round(norm_c1_gain, 4),
            'S_C1_syn': round(sc1_syn, 4),
            'S_C2_Score': round(norm_common_act, 4),
            'S_C3_Score': round(norm_bad_rev, 4),
            'S_C4_Score': round(norm_opp, 4),
            'S_C5_Score': round(norm_wrong_dir, 4),
            'S_C6_Coverage': round(coverage_score_c6, 4),
            'C6_Hit_Detail': c6_coverage_detail,
            'C1_Gain_Detail': ";".join(list_success) if list_success else "None",
            'C2_Common_Detail': ";".join(list_common_act) if list_common_act else "None",
            'C3_Pathoge_Detail': ";".join(list_bad_rev) if list_bad_rev else "None",
            'C4_Conflict_Detail': ";".join(list_conflict) if list_conflict else "None",
            'C5_WrongDir_Detail': ";".join(list_wrong_dir) if list_wrong_dir else "None"
        }

    # =================================================================
    # 4. Phase-wise Virtual Screening Executions
    # =================================================================
    all_drugs_list = sorted(drug_targets.keys())

    print(f"[PHASE I] Evaluating baseline monotherapy virtual screening metrics (N={len(all_drugs_list)})...")
    single_results = [calculate_score((d,)) for d in all_drugs_list]

    single_c1_lookup = {res['Drugs']: res['S_C1_Score'] for res in single_results}
    single_prs_lookup = {res['Drugs']: res['PRS'] for res in single_results}

    # Restrict combinatorics pool to Top N high-scoring monotherapies
    df_singles = pd.DataFrame(single_results).sort_values('PRS', ascending=False)
    top_n = min(TOP_SINGLE_N, len(all_drugs_list))
    top_drugs = df_singles['Drugs'].head(top_n).tolist()

    print(f"\n[PHASE II] Initiating higher-order combinatorial network synergy filtering on Top {top_n} candidates...")
    combo_results = []

    for k in [2, 3, 4]:
        combos = list(combinations(top_drugs, k))

        # Restrict computation arrays to combinations mapping onto at least one primary phenotype indicator
        valid_combos = []
        if len(target_diseases) > 0:
            for c in combos:
                c_dis = set()
                for d in c: c_dis.update(drug_to_diseases.get(d.lower(), set()))
                if len(target_diseases.intersection(c_dis)) > 0:
                    valid_combos.append(c)
        else:
            valid_combos = combos

        print(f"  > Profiling {k}-agent therapeutic matrices: Initial combinations n={len(combos)}")
        print(f"  > Active candidates filtered by clinical phenotype inclusion (Coverage > 0%) n={len(valid_combos)}")

        for idx, c in enumerate(valid_combos):
            if (idx + 1) % 5000 == 0:
                print(f"    Processing iteration: {idx + 1} candidates computed...")
            combo_results.append(
                calculate_score(c, single_c1_lookup=single_c1_lookup, single_prs_lookup=single_prs_lookup)
            )

    # =================================================================
    # 5. Min-Max Normalization, Sort Ordering & File Writing
    # =================================================================
    df_final = pd.DataFrame(single_results + combo_results)

    if not df_final.empty:
        prs_min = df_final['PRS'].min()
        prs_max = df_final['PRS'].max()
        if prs_max > prs_min:
            df_final['PRS_norm'] = (df_final['PRS'] - prs_min) / (prs_max - prs_min)
            df_final['PRS_norm'] = df_final['PRS_norm'].round(4)
        else:
            df_final['PRS_norm'] = 1.0
    else:
        df_final['PRS_norm'] = None

    df_final = df_final.sort_values('PRS', ascending=False)

    cols = ['Drugs', 'Disease_Coverage', 'Coverage_Ratio', 'PRS', 'PRS_norm', 'PRS_syn',
            'S_C1_Score', 'S_C1_syn', 'S_C2_Score', 'S_C3_Score', 'S_C4_Score', 'S_C5_Score', 'S_C6_Coverage',
            'C6_Hit_Detail', 'C1_Gain_Detail', 'C2_Common_Detail', 'C3_Pathoge_Detail', 'C4_Conflict_Detail',
            'C5_WrongDir_Detail']

    df_final = df_final[[c for c in cols if c in df_final.columns]]

    df_final.to_csv(args.output, index=False)
    print(f"\n[SUCCESS] Matrix computation finalized. Unified screening metrics exported to: {args.output}")


if __name__ == "__main__":
    main()