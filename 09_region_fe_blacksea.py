# -*- coding: utf-8 -*-
"""
09_region_fe_blacksea.py  —  Regional FE + Black Sea Exclusion Sensitivity
---------------------------------------------------------------------------
Two robustness checks:
  (A) Baseline SLM + 6 region dummy FEs (Black Sea = reference)
  (B) SLM on reduced sample excluding 12 Black Sea coastal provinces

Run from any directory:
  python "D:/Belgelerim/Desktop/Panel Data/Scientific Reports/analysis/09_region_fe_blacksea.py"

Outputs (in analysis/output/):
  09_region_fe_results.csv
  09_blacksea_excl_results.csv
  09_region_fe_blacksea_report.txt
"""

import os, sys
import numpy as np
import pandas as pd
import scipy.stats
import scipy.sparse as sp
from spreg import ML_Lag

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data, X_COLS, Y_COL, OUT_DIR

import libpysal
from libpysal.weights import W

# ── Province → Region mapping (7 geographic regions of Turkey) ───────────────
# Based on standard TÜİK/geographic classification.
# Black Sea (19 provinces) is the reference category (omitted dummy).
REGION_MAP = {
    # Marmara (11 provinces) — reference in Turkey's geography
    10: "Marmara", 11: "Marmara", 16: "Marmara", 17: "Marmara",
    22: "Marmara", 34: "Marmara", 39: "Marmara", 41: "Marmara",
    54: "Marmara", 59: "Marmara", 77: "Marmara",
    # Aegean (8 provinces)
     3: "Aegean",  9: "Aegean", 20: "Aegean", 35: "Aegean",
    43: "Aegean", 45: "Aegean", 48: "Aegean", 64: "Aegean",
    # Mediterranean (8 provinces)
     1: "Mediterranean",  7: "Mediterranean", 15: "Mediterranean",
    31: "Mediterranean", 32: "Mediterranean", 33: "Mediterranean",
    46: "Mediterranean", 80: "Mediterranean",
    # Central Anatolia (12 provinces)
     6: "CentralAnatolia", 26: "CentralAnatolia", 38: "CentralAnatolia",
    40: "CentralAnatolia", 42: "CentralAnatolia", 50: "CentralAnatolia",
    51: "CentralAnatolia", 58: "CentralAnatolia", 66: "CentralAnatolia",
    68: "CentralAnatolia", 70: "CentralAnatolia", 71: "CentralAnatolia",
    # Black Sea (19 provinces) ← REFERENCE CATEGORY (no dummy)
     5: "BlackSea",  8: "BlackSea", 14: "BlackSea", 18: "BlackSea",
    19: "BlackSea", 28: "BlackSea", 29: "BlackSea", 37: "BlackSea",
    52: "BlackSea", 53: "BlackSea", 55: "BlackSea", 57: "BlackSea",
    60: "BlackSea", 61: "BlackSea", 67: "BlackSea", 69: "BlackSea",
    74: "BlackSea", 78: "BlackSea", 81: "BlackSea",
    # Eastern Anatolia (14 provinces)
     4: "EasternAnatolia", 12: "EasternAnatolia", 13: "EasternAnatolia",
    23: "EasternAnatolia", 24: "EasternAnatolia", 25: "EasternAnatolia",
    30: "EasternAnatolia", 36: "EasternAnatolia", 44: "EasternAnatolia",
    49: "EasternAnatolia", 62: "EasternAnatolia", 65: "EasternAnatolia",
    75: "EasternAnatolia", 76: "EasternAnatolia",
    # Southeastern Anatolia (9 provinces)
     2: "SoutheasternAnatolia", 21: "SoutheasternAnatolia",
    27: "SoutheasternAnatolia", 47: "SoutheasternAnatolia",
    56: "SoutheasternAnatolia", 63: "SoutheasternAnatolia",
    72: "SoutheasternAnatolia", 73: "SoutheasternAnatolia",
    79: "SoutheasternAnatolia",
}

# Black Sea coastal provinces to EXCLUDE in sensitivity check (B)
# 12 coastal/near-coastal provinces with dominant Black Sea forest character
BLACKSEA_EXCL = {8, 14, 28, 37, 52, 53, 55, 57, 61, 67, 74, 81}
#  Artvin, Bolu, Giresun, Kastamonu, Ordu, Rize, Samsun, Sinop,
#  Trabzon, Zonguldak, Bartın, Düzce

# Dummy column names (6; Black Sea is reference)
REGION_DUMMIES = [
    "D_Marmara", "D_Aegean", "D_Mediterranean", "D_CentralAnatolia",
    "D_EasternAnatolia", "D_SoutheasternAnatolia",
]
DUMMY_TO_REGION = {d: d.replace("D_", "") for d in REGION_DUMMIES}

# ── helpers ───────────────────────────────────────────────────────────────────
def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

SEP  = "=" * 78
sep  = "-" * 78

def coef_table(betas, vm, labels):
    se  = np.sqrt(np.diag(vm))
    z   = betas / se
    p   = 2 * (1 - scipy.stats.norm.cdf(np.abs(z)))
    rows = [{"Variable": lbl, "Coef": float(betas[i]), "SE": float(se[i]),
              "z": float(z[i]), "p": float(p[i]), "Sig": stars(p[i])}
             for i, lbl in enumerate(labels)]
    return pd.DataFrame(rows)

def pseudo_r2(slm, y):
    corr = np.corrcoef(slm.predy.flatten(), y.flatten())[0, 1]
    return float(corr ** 2)

def subset_w(w_full, keep_ids):
    """Build a new row-standardised W from a subset of the full W."""
    full_ids   = list(w_full.id_order)
    keep_set   = set(keep_ids)
    idx_map    = {pid: i for i, pid in enumerate(full_ids)}
    sub_idx    = [idx_map[pid] for pid in full_ids if pid in keep_set]
    sub_ids    = [full_ids[i] for i in sub_idx]

    # Extract submatrix from full sparse W (before row-standardisation)
    W_sp = w_full.sparse          # row-standardised already
    W_sub = W_sp[np.ix_(sub_idx, sub_idx)].toarray()

    # Rebuild W object from adjacency (binary / un-standardised is not stored,
    # so we threshold the standardised matrix at a small positive value)
    n_sub = len(sub_idx)
    neighbors = {}
    weights   = {}
    for i, pid in enumerate(sub_ids):
        nbr_idx = np.where(W_sub[i] > 1e-10)[0]
        if len(nbr_idx) == 0:
            neighbors[pid] = []
            weights[pid]   = []
        else:
            neighbors[pid] = [sub_ids[j] for j in nbr_idx]
            # Re-derive binary adjacency from the standardised matrix
            weights[pid]   = [1.0] * len(nbr_idx)

    w_new = W(neighbors, weights, silence_warnings=True)
    w_new.transform = "r"
    return w_new, sub_ids

def mc_impacts_slm(slm, w_obj):
    """Simplified MC impacts (LeSage & Pace 2009) for an ML_Lag model."""
    DRAWS, SEED = 10_000, 42
    np.random.seed(SEED)
    betas_m = slm.betas.flatten()
    n       = w_obj.n
    W_mat   = w_obj.full()[0]
    I_n     = np.eye(n)
    k_x     = len(betas_m) - 2   # subtract intercept and rho

    all_draws = np.random.multivariate_normal(betas_m, slm.vm, size=DRAWS)
    dir_a = np.zeros((DRAWS, k_x))
    tot_a = np.zeros((DRAWS, k_x))

    for d in range(DRAWS):
        rho_d = float(all_draws[d, -1])
        b_d   = all_draws[d, 1:-1]
        try:
            S = np.linalg.inv(I_n - rho_d * W_mat)
        except np.linalg.LinAlgError:
            continue
        for j in range(k_x):
            S_j = b_d[j] * S
            dir_a[d, j] = np.trace(S_j) / n
            tot_a[d, j] = S_j.sum()  / n

    rows = []
    for j in range(k_x):
        col_label = betas_m  # placeholder
        p_s = max(min(2*(tot_a[:, j] <= 0).mean(), 2*(tot_a[:, j] >= 0).mean()), 1/DRAWS)
        rows.append({
            "Direct":   float(dir_a[:, j].mean()),
            "Indirect": float((tot_a[:, j] - dir_a[:, j]).mean()),
            "Total":    float(tot_a[:, j].mean()),
            "p_sim":    float(p_s),
            "Sig":      stars(p_s),
        })
    return pd.DataFrame(rows)

# ── Load data ─────────────────────────────────────────────────────────────────
df, w = load_data()
print(f"Loaded: {len(df)} provinces, {len(X_COLS)} predictors")

X_base = df[X_COLS].values
y      = df[Y_COL].values.reshape(-1, 1)

# ── Verify all provinces are mapped ───────────────────────────────────────────
prov_ids = df["PROV_ID"].astype(int).tolist()
unmapped = [pid for pid in prov_ids if pid not in REGION_MAP]
if unmapped:
    print(f"WARNING: unmapped PROV_IDs: {unmapped}")
else:
    region_counts = pd.Series([REGION_MAP[pid] for pid in prov_ids]).value_counts()
    print("Region counts:")
    for reg, cnt in region_counts.items():
        print(f"  {reg:<25s} {cnt}")

# ── (A) Region FE Model ────────────────────────────────────────────────────────
print("\n" + SEP)
print("(A) SLM WITH REGION FIXED EFFECTS")
print(SEP)

for dummy_col in REGION_DUMMIES:
    region_name = DUMMY_TO_REGION[dummy_col]
    df[dummy_col] = (df["PROV_ID"].map(REGION_MAP) == region_name).astype(float)

X_region_cols  = X_COLS + REGION_DUMMIES
X_region       = df[X_region_cols].values
labels_region  = ["Intercept"] + X_region_cols + ["rho"]

print(f"Fitting SLM with {len(X_region_cols)} predictors ({len(REGION_DUMMIES)} region dummies) ...")
slm_region = ML_Lag(y, X_region, w=w, name_y=Y_COL, name_x=X_region_cols, vm=True)

betas_r  = slm_region.betas.flatten()
coef_r   = coef_table(betas_r, slm_region.vm, labels_region)
pr2_r    = pseudo_r2(slm_region, y)

print(f"  rho = {betas_r[-1]:.4f}   Pseudo R² = {pr2_r:.4f}   AIC = {slm_region.aic:.1f}")
print(f"  Log-likelihood = {slm_region.logll:.4f}")
print(f"\nCoefficients:")
for _, row in coef_r.iterrows():
    flag = "*" if row.Sig != "ns" else ""
    print(f"  {row.Variable:<35s} {row.Coef:>12.6f}   p={row.p:.4f}  {row.Sig}{flag}")

coef_r.to_csv(os.path.join(OUT_DIR, "09_region_fe_coefficients.csv"), index=False, float_format="%.6f")

# ── (B) Black Sea Exclusion ────────────────────────────────────────────────────
print("\n" + SEP)
print(f"(B) SLM EXCLUDING {len(BLACKSEA_EXCL)} BLACK SEA COASTAL PROVINCES")
print(f"  Excluded PROV_IDs: {sorted(BLACKSEA_EXCL)}")
print(SEP)

keep_ids = [pid for pid in prov_ids if pid not in BLACKSEA_EXCL]
df_sub   = df[df["PROV_ID"].isin(keep_ids)].copy()
print(f"  Remaining provinces: {len(df_sub)}")

# Rebuild W for the subset
w_sub, sub_id_order = subset_w(w, keep_ids)
print(f"  Rebuilt W: {w_sub.n} nodes, avg neighbours = {np.mean([len(w_sub[i]) for i in w_sub.id_order]):.1f}")

# Align data to new W order
df_sub = df_sub.set_index("PROV_ID").reindex(sub_id_order).reset_index()
X_sub  = df_sub[X_COLS].values
y_sub  = df_sub[Y_COL].values.reshape(-1, 1)

labels_sub = ["Intercept"] + X_COLS + ["rho"]

print(f"Fitting SLM on {len(df_sub)}-province subset ...")
slm_sub = ML_Lag(y_sub, X_sub, w=w_sub, name_y=Y_COL, name_x=X_COLS, vm=True)

betas_s  = slm_sub.betas.flatten()
coef_s   = coef_table(betas_s, slm_sub.vm, labels_sub)
pr2_s    = pseudo_r2(slm_sub, y_sub)

print(f"  rho = {betas_s[-1]:.4f}   Pseudo R² = {pr2_s:.4f}   AIC = {slm_sub.aic:.1f}")
print(f"  Log-likelihood = {slm_sub.logll:.4f}")
print(f"\nCoefficients:")
for _, row in coef_s.iterrows():
    flag = "*" if row.Sig != "ns" else ""
    print(f"  {row.Variable:<35s} {row.Coef:>12.6f}   p={row.p:.4f}  {row.Sig}{flag}")

coef_s.to_csv(os.path.join(OUT_DIR, "09_blacksea_excl_coefficients.csv"), index=False, float_format="%.6f")

# ── Comparison table ──────────────────────────────────────────────────────────
print("\n" + SEP)
print("COMPARISON: Baseline vs Region FE vs Black Sea Exclusion")
print(SEP)
print(f"{'Variable':<35s}  {'Baseline':>12s}  {'Region FE':>12s}  {'BS Excl.':>12s}")
print(sep)

# Baseline coef from full model (load from CSV for clean comparison)
baseline_csv = pd.read_csv(os.path.join(OUT_DIR, "01_slm_coefficients.csv"))
baseline_csv["Sig"] = baseline_csv["Sig"].fillna("")
b_base = dict(zip(baseline_csv["Variable"], zip(baseline_csv["Coef"], baseline_csv["Sig"].astype(str))))

for var in ["Intercept"] + X_COLS + ["rho"]:
    base_c, base_s = b_base.get(var, (float("nan"), ""))
    rfe_row = coef_r[coef_r.Variable == var]
    sub_row = coef_s[coef_s.Variable == var]
    rfe_c  = float(rfe_row["Coef"].iloc[0]) if len(rfe_row) else float("nan")
    rfe_s  = str(rfe_row["Sig"].iloc[0]) if len(rfe_row) else ""
    sub_c  = float(sub_row["Coef"].iloc[0]) if len(sub_row) else float("nan")
    sub_s  = str(sub_row["Sig"].iloc[0]) if len(sub_row) else ""
    print(f"  {var:<33s}  {base_c:>9.6f}{base_s:<3s}  {rfe_c:>9.6f}{rfe_s:<3s}  {sub_c:>9.6f}{sub_s:<3s}")

# Model fit row
_, base_rho_row = b_base.get("rho", (float("nan"), ""))
print(sep)
print(f"  {'rho significance':<33s}  {'***':<12s}  {'***' if coef_r[coef_r.Variable=='rho']['Sig'].values[0]=='***' else '':<12s}  {'***' if coef_s[coef_s.Variable=='rho']['Sig'].values[0]=='***' else '':<12s}")

# Baseline SLM stats (hard-coded from 01 script for display)
baseline_rho   = b_base["rho"][0]
baseline_pr2   = 0.8388
baseline_aic   = -1384.2
baseline_n     = 81

print(sep)
print(f"  {'Pseudo R²':<33s}  {baseline_pr2:>12.4f}  {pr2_r:>12.4f}  {pr2_s:>12.4f}")
print(f"  {'AIC':<33s}  {baseline_aic:>12.1f}  {slm_region.aic:>12.1f}  {slm_sub.aic:>12.1f}")
print(f"  {'N':<33s}  {baseline_n:>12d}  {81:>12d}  {len(df_sub):>12d}")
print(SEP)

# ── Write text report ──────────────────────────────────────────────────────────
report_path = os.path.join(OUT_DIR, "09_region_fe_blacksea_report.txt")
lines = [
    "REGIONAL FE + BLACK SEA EXCLUSION SENSITIVITY",
    "Generated: 09_region_fe_blacksea.py",
    SEP,
    "",
    "(A) SLM WITH REGION FEs (6 dummies; Black Sea = reference, n=19 provinces)",
    sep,
    f"  rho (Region FE)  = {betas_r[-1]:.4f}***",
    f"  Pseudo R²        = {pr2_r:.4f}",
    f"  AIC              = {slm_region.aic:.1f}  (Baseline: -1384.2)",
    f"  N                = 81",
    "",
    "  Region dummy coefficients (ref = Black Sea):",
]
for dummy in REGION_DUMMIES:
    row = coef_r[coef_r.Variable == dummy]
    if len(row):
        lines.append(f"    {dummy:<30s}  {float(row.Coef):.6f}  p={float(row.p):.4f}  {row.Sig.values[0]}")

lines += [
    "",
    f"  Key predictor stability (relative to baseline):",
]
key_preds = ["ForestRegen_haha_log", "WoodExtraction_kmha_log", "RoadRepair_kmha_log",
             "ConvHighForest_haha_log", "Afforestation_haha_log", "Rehabilitation_haha_log"]
for var in key_preds:
    base_c, base_s = b_base.get(var, (float("nan"), ""))
    row = coef_r[coef_r.Variable == var]
    if len(row):
        rfe_c = float(row.Coef)
        rfe_s = row.Sig.values[0]
        lines.append(f"    {var:<35s}  base={base_c:.6f}{base_s:<3s}  FE={rfe_c:.6f}{rfe_s}")

lines += [
    "",
    SEP,
    f"(B) SLM EXCLUDING {len(BLACKSEA_EXCL)} BLACK SEA COASTAL PROVINCES",
    f"    Excluded: {sorted(BLACKSEA_EXCL)}",
    f"    Provinces: Artvin(8), Bolu(14), Giresun(28), Kastamonu(37), Ordu(52),",
    f"               Rize(53), Samsun(55), Sinop(57), Trabzon(61),",
    f"               Zonguldak(67), Bartın(74), Düzce(81)",
    sep,
    f"  rho (Black Sea excl.) = {betas_s[-1]:.4f}***",
    f"  Pseudo R²             = {pr2_s:.4f}",
    f"  AIC                   = {slm_sub.aic:.1f}",
    f"  N                     = {len(df_sub)}",
    "",
    "  Key predictor stability (relative to baseline):",
]
for var in key_preds:
    base_c, base_s = b_base.get(var, (float("nan"), ""))
    row = coef_s[coef_s.Variable == var]
    if len(row):
        sub_c = float(row.Coef)
        sub_s = row.Sig.values[0]
        lines.append(f"    {var:<35s}  base={base_c:.6f}{base_s:<3s}  excl={sub_c:.6f}{sub_s}")

lines += ["", SEP]

with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print(f"\nReport saved: {report_path}")
