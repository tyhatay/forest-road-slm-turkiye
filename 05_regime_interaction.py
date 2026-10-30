# -*- coding: utf-8 -*-
"""
05_regime_interaction.py  —  Regime Heterogeneity via Interaction Terms
-----------------------------------------------------------------------
Addresses Reviewer 2, Major Comment 5:
  "The spatial regimes analysis is saturated (n≈10 HH provinces,
   ~13 parameters per regime). Please replace with a pooled SLM
   augmented by interaction terms."

Strategy:
  1. Replicate LISA clusters (Moran_Local on Y, p<0.05, geoda_quads)
     → 10 HH provinces (Black Sea production corridor)
  2. Create binary HH_dummy (1 = HH province, 0 = others)
  3. Augment SLM: add HH_dummy + HH × Uretim + HH × Yol_bakimi
     (only the two most operationally significant variables)
  4. LR test vs baseline SLM (H0: all interaction terms = 0, df = 3)
  5. Wald z-tests for individual interaction terms

Old regime approach: 3 regimes × 13 params = 39 params, n=10 for HH → df<0
New approach: pooled + 3 terms (dummy + 2 interactions) → df=81-15=66

Run:
  conda run -n gis python 05_regime_interaction.py

Outputs (in analysis/output/):
  05_regime_interaction_results.txt
  05_regime_interaction_coefficients.csv
  05_lisa_clusters.csv
"""

import os, sys
import numpy as np
import pandas as pd
import scipy.stats
from spreg import ML_Lag
from esda import Moran_Local
import libpysal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data, X_COLS, Y_COL, OUT_DIR

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

# ── Load data ──────────────────────────────────────────────────────────────────
df, w = load_data()
print(f"Data: {len(df)} provinces")

X = df[X_COLS].values
y = df[Y_COL].values.reshape(-1, 1)

# ── Step 1: LISA clusters on Y ────────────────────────────────────────────────
SEED = 42
y_lisa = df[Y_COL].values
lisa   = Moran_Local(y_lisa, w, permutations=999, geoda_quads=True, seed=SEED)

# geoda_quads=True: q=1=HH, q=2=LL, q=3=LH, q=4=HL; p_sim < 0.05 → significant
clusters = np.where(lisa.p_sim < 0.05, lisa.q, 0)   # 0 = not significant
df["LISA_q"] = clusters

cluster_counts = pd.Series(clusters).value_counts().sort_index()
print("\nLISA cluster distribution (q; 0=NS):")
for q_val, cnt in cluster_counts.items():
    label = {0:"NS (not significant)", 1:"HH (high-high)",
             2:"LL (low-low)", 3:"LH (low-high)", 4:"HL (high-low)"}.get(q_val, str(q_val))
    print(f"  q={q_val}  {label}: {cnt} provinces")

# HH dummy: 1 = HH province, 0 = all others
df["HH_dummy"] = (clusters == 1).astype(float)
n_HH = int(df["HH_dummy"].sum())
print(f"\nHH provinces: {n_HH}  |  Non-HH: {81 - n_HH}")

# Save cluster assignments
cluster_df = df[["PROV_ID","LISA_q","HH_dummy"]].copy()
cluster_df.to_csv(os.path.join(OUT_DIR, "05_lisa_clusters.csv"), index=False)

# ── Step 2: Baseline SLM (replication) ────────────────────────────────────────
print("\nFitting baseline SLM ...")
slm_base = ML_Lag(y, X, w=w, name_y=Y_COL, name_x=X_COLS, vm=True)
LL_base  = float(slm_base.logll)
AIC_base = float(slm_base.aic)
BIC_base = float(slm_base.schwarz)
rho_base = float(slm_base.rho)
k_base   = len(X_COLS)   # X params (excl. intercept and rho)
print(f"  LL={LL_base:.4f}  AIC={AIC_base:.4f}  BIC={BIC_base:.4f}  rho={rho_base:.5f}")

# ── Step 3: Interaction SLM ───────────────────────────────────────────────────
# Add: HH_dummy + HH × Uretim_tali + HH × Yol_bakimi
HH     = df["HH_dummy"].values
X_uret = df["WoodExtraction_kmha_log"].values
X_yol  = df["RoadRepair_kmha_log"].values

# Interaction columns
HH_Uret = HH * X_uret
HH_Yol  = HH * X_yol

X_int    = np.column_stack([X, HH, HH_Uret, HH_Yol])
XCOLS_int = X_COLS + ["HH_dummy", "HH_x_WoodExtraction", "HH_x_RoadRepair"]

print("\nFitting interaction SLM (X + HH_dummy + HH×WoodExtraction + HH×RoadRepair) ...")
slm_int = ML_Lag(y, X_int, w=w, name_y=Y_COL, name_x=XCOLS_int, vm=True)
LL_int  = float(slm_int.logll)
AIC_int = float(slm_int.aic)
BIC_int = float(slm_int.schwarz)
rho_int = float(slm_int.rho)
print(f"  LL={LL_int:.4f}  AIC={AIC_int:.4f}  BIC={BIC_int:.4f}  rho={rho_int:.5f}")

# ── Step 4: LR test (H0: interaction terms jointly = 0) ─────────────────────
n_int_terms = 3   # HH_dummy + 2 interactions
LR_stat  = 2.0 * (LL_int - LL_base)
LR_p     = scipy.stats.chi2.sf(LR_stat, n_int_terms)
delta_BIC = BIC_int - BIC_base
delta_AIC = AIC_int - AIC_base

print(f"\nLR test H0: HH terms jointly = 0")
print(f"  LR = {LR_stat:.4f}  df = {n_int_terms}  p = {LR_p:.6f}  {stars(LR_p)}")
print(f"  ΔBIC = {delta_BIC:+.4f}  ΔAIC = {delta_AIC:+.4f}")

# ── Step 5: Coefficient table ─────────────────────────────────────────────────
betas  = slm_int.betas.flatten()
se_all = np.sqrt(np.diag(slm_int.vm))
z_all  = betas / se_all
p_all  = 2 * (1 - scipy.stats.norm.cdf(np.abs(z_all)))
labels = ["Intercept"] + XCOLS_int + ["rho"]

coef_rows = [
    {"Variable": labels[i], "Coef": float(betas[i]),
     "SE": float(se_all[i]), "z": float(z_all[i]),
     "p": float(p_all[i]), "Sig": stars(p_all[i])}
    for i in range(len(betas))
]
coef_df = pd.DataFrame(coef_rows)
coef_df.to_csv(os.path.join(OUT_DIR, "05_regime_interaction_coefficients.csv"),
               index=False, float_format="%.6f")

# ── Marginal effects for HH provinces ─────────────────────────────────────────
# Intercept adj for HH: beta_HH_dummy
# WoodExtraction total effect in HH: beta_WoodExtraction + beta_HH_x_WoodExtraction
# RoadRepair total effect in HH:     beta_RoadRepair + beta_HH_x_RoadRepair
idx_HH    = labels.index("HH_dummy")
idx_HHU   = labels.index("HH_x_WoodExtraction")
idx_HHY   = labels.index("HH_x_RoadRepair")
idx_Uret  = labels.index("WoodExtraction_kmha_log")
idx_Yol   = labels.index("RoadRepair_kmha_log")

b_HH    = betas[idx_HH]
b_HHU   = betas[idx_HHU]
b_HHY   = betas[idx_HHY]
b_Uret  = betas[idx_Uret]
b_Yol   = betas[idx_Yol]

# Effective coefficients
eff_Uret_nonHH = b_Uret
eff_Uret_HH    = b_Uret + b_HHU
eff_Yol_nonHH  = b_Yol
eff_Yol_HH     = b_Yol + b_HHY

# ── Report ─────────────────────────────────────────────────────────────────────
SEP = "=" * 74
sep = "-" * 74

lines = [
    SEP,
    "REGIME HETEROGENEITY — INTERACTION TERMS APPROACH",
    "Reviewer 2, Major Comment 5",
    "  Old approach: ML_Lag_Regimes, n_HH≈10, params≈13 → saturated (df≈−3)",
    "  New approach: pooled SLM + HH_dummy + 2 interactions → df=81−15=66",
    SEP,
    "",
    "LISA CLUSTER SUMMARY  (Moran_Local, p<0.05, 999 permutations, seed=42)",
    sep,
]
for q_val, cnt in cluster_counts.items():
    label = {0:"NS", 1:"HH", 2:"LL", 3:"LH", 4:"HL"}.get(q_val, str(q_val))
    lines.append(f"  q={q_val}  {label}: {cnt} provinces")
lines += [
    f"  → HH dummy: {n_HH} provinces assigned 1 (Black Sea production corridor)",
    "",
    "MODEL COMPARISON",
    sep,
    f"  {'Model':35s} {'LL':>10} {'AIC':>10} {'BIC':>10} {'n_par':>6} {'rho':>8}",
    sep,
    f"  {'Baseline SLM':35s} {LL_base:10.4f} {AIC_base:10.4f} {BIC_base:10.4f} "
    f"{k_base+2:6d} {rho_base:8.5f}",
    f"  {'Interaction SLM (+3 terms)':35s} {LL_int:10.4f} {AIC_int:10.4f} {BIC_int:10.4f} "
    f"{k_base+2+n_int_terms:6d} {rho_int:8.5f}",
    sep,
    f"  ΔAIC (Int − Base): {delta_AIC:+.4f}  ({'Int preferred by AIC' if delta_AIC<0 else 'Base preferred by AIC'})",
    f"  ΔBIC (Int − Base): {delta_BIC:+.4f}  ({'Int preferred by BIC' if delta_BIC<0 else 'Base preferred by BIC — parsimony favours pooled SLM'})",
    "",
    "LR TEST  H0: HH_dummy + HH×WoodExtraction + HH×RoadRepair jointly = 0",
    sep,
    f"  LR statistic: {LR_stat:.4f}",
    f"  df          : {n_int_terms}",
    f"  p-value     : {LR_p:.6f}  {stars(LR_p)}",
    f"  chi2({n_int_terms}) critical:  5%={scipy.stats.chi2.ppf(0.95,n_int_terms):.3f}  "
    f"1%={scipy.stats.chi2.ppf(0.99,n_int_terms):.3f}",
    "",
    "INTERACTION SLM — FULL COEFFICIENT TABLE",
    sep,
    f"  {'Variable':42s} {'Coef':>12} {'SE':>10} {'p':>8}  Sig",
    sep,
]
for _, r in coef_df.iterrows():
    lines.append(
        f"  {r['Variable']:42s} {r['Coef']:12.6f} {r['SE']:10.6f} {r['p']:8.4f}  {r['Sig']}"
    )

lines += [
    sep,
    "",
    "MARGINAL EFFECTS BY REGIME",
    sep,
    f"  {'Variable':30s} {'Non-HH β':>12} {'HH β':>12} {'HH excess β':>14} {'sig_excess':>10}",
    sep,
    f"  {'Wood Extraction roads':30s} {eff_Uret_nonHH:12.6f} {eff_Uret_HH:12.6f} "
    f"{b_HHU:14.6f} {stars(p_all[idx_HHU]):>10}",
    f"  {'Road Repair intensity':30s} {eff_Yol_nonHH:12.6f} {eff_Yol_HH:12.6f} "
    f"{b_HHY:14.6f} {stars(p_all[idx_HHY]):>10}",
    sep,
    "",
    "INTERPRETATION FOR REVISION RESPONSE:",
]

if LR_p < 0.05:
    lines += [
        f"  The LR test rejects H0 (p={LR_p:.4f}): HH-regime heterogeneity is",
        f"  statistically significant. Provinces in the Black Sea production",
        f"  corridor exhibit different sensitivity to operational drivers.",
        f"  However, ΔBIC={delta_BIC:+.2f} — parsimony may still favour the pooled SLM.",
        f"  The interaction approach is appropriate: it avoids the saturation",
        f"  problem of the old regime model while capturing regime differences.",
    ]
else:
    lines += [
        f"  The LR test fails to reject H0 (p={LR_p:.4f}): the HH-regime",
        f"  differences in WoodExtraction and RoadRepair are not statistically",
        f"  distinguishable from zero. The pooled SLM is adequate.",
        f"  This is consistent with the main finding: the demand-pull mechanism",
        f"  operates similarly across spatial regimes.",
    ]

lines += [
    f"  Note: The BIC criterion ({delta_BIC:+.2f}) {'penalises' if delta_BIC>0 else 'favours'} the interaction model.",
    "",
    SEP,
]

report = "\n".join(lines)
out_txt = os.path.join(OUT_DIR, "05_regime_interaction_results.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(report)

print("\n" + report)
print(f"\nSaved: {out_txt}")
