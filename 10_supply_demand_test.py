# -*- coding: utf-8 -*-
"""
10_supply_demand_test.py  —  Joint Supply-vs-Demand Test
---------------------------------------------------------
Tests H0: all 10 operational coefficients jointly = 0
  (i.e., growing stock density alone explains road construction intensity)

Two equivalent approaches:
  (LR)   Likelihood Ratio test: restricted (supply only) vs unrestricted (baseline)
  (Wald) Wald chi-square from the unrestricted model VM matrix

Run:
  python "D:/Belgelerim/Desktop/Panel Data/Scientific Reports/analysis/10_supply_demand_test.py"

Outputs (in analysis/output/):
  10_supply_demand_report.txt
"""

import os, sys
import numpy as np
import pandas as pd
import scipy.stats
from spreg import ML_Lag

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data, X_COLS, Y_COL, OUT_DIR

# Supply proxy (1 predictor)
SUPPLY_COL = "GrowingStock_m3ha_log"

# Operational (demand) indicators (10 predictors — everything except supply)
DEMAND_COLS = [c for c in X_COLS if c != SUPPLY_COL]

SEP = "=" * 70
sep = "-" * 70

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

# ── Load data ─────────────────────────────────────────────────────────────────
df, w = load_data()
y = df[Y_COL].values.reshape(-1, 1)

# ── Unrestricted model (baseline SLM, all 11 predictors) ─────────────────────
print("Fitting unrestricted SLM (all 11 predictors) ...")
X_full  = df[X_COLS].values
slm_u   = ML_Lag(y, X_full, w=w, name_y=Y_COL, name_x=X_COLS, vm=True)
ll_u    = float(slm_u.logll)
k_u     = len(X_COLS) + 2          # k predictors + intercept + rho
print(f"  logLL = {ll_u:.4f}   AIC = {slm_u.aic:.1f}   k = {k_u}")

# ── Restricted model (supply-only SLM, 1 predictor) ──────────────────────────
print("Fitting restricted SLM (supply only: GrowingStock) ...")
X_restricted = df[[SUPPLY_COL]].values
slm_r        = ML_Lag(y, X_restricted, w=w, name_y=Y_COL,
                      name_x=[SUPPLY_COL], vm=True)
ll_r  = float(slm_r.logll)
k_r   = 1 + 2                      # 1 predictor + intercept + rho
print(f"  logLL = {ll_r:.4f}   AIC = {slm_r.aic:.1f}   k = {k_r}")

# ── LR test ───────────────────────────────────────────────────────────────────
df_lr  = len(DEMAND_COLS)          # = 10
LR_stat = 2.0 * (ll_u - ll_r)
p_lr    = scipy.stats.chi2.sf(LR_stat, df=df_lr)

print(f"\nLR test: LR = {LR_stat:.2f}, df = {df_lr}, p = {p_lr:.6f}  {stars(p_lr)}")

# ── Wald test ─────────────────────────────────────────────────────────────────
# betas_u layout: [intercept, β_0..β_10, rho]  → demand indices = 2..11 (1-based)
# Intercept is index 0; SUPPLY is index 1; DEMAND are indices 2..11
betas_u  = slm_u.betas.flatten()
vm_u     = slm_u.vm

demand_idx = [i + 1 for i, col in enumerate(X_COLS) if col in DEMAND_COLS]  # +1 for intercept offset
# demand_idx should be [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

beta_d   = betas_u[demand_idx]
vm_d     = vm_u[np.ix_(demand_idx, demand_idx)]
W_wald   = float(beta_d @ np.linalg.inv(vm_d) @ beta_d)
p_wald   = scipy.stats.chi2.sf(W_wald, df=len(demand_idx))

print(f"Wald test: W = {W_wald:.2f}, df = {len(demand_idx)}, p = {p_wald:.6f}  {stars(p_wald)}")

# ── Restricted model coefficient summary ─────────────────────────────────────
betas_r  = slm_r.betas.flatten()
se_r     = np.sqrt(np.diag(slm_r.vm))
z_r      = betas_r / se_r
p_r      = 2 * (1 - scipy.stats.norm.cdf(np.abs(z_r)))
labels_r = ["Intercept", SUPPLY_COL, "rho"]

print(f"\nRestricted model (supply only):")
for i, lbl in enumerate(labels_r):
    print(f"  {lbl:<35s}  {betas_r[i]:>12.6f}   p={p_r[i]:.4f}  {stars(p_r[i])}")
print(f"  Pseudo R²  = {np.corrcoef(slm_r.predy.flatten(), y.flatten())[0,1]**2:.4f}")

# ── Report ────────────────────────────────────────────────────────────────────
lines = [
    "JOINT SUPPLY-vs-DEMAND TEST  (SLM, n=81)",
    "Generated: 10_supply_demand_test.py",
    SEP,
    "",
    "Framing:",
    "  Supply proxy   : GrowingStock_m3ha_log (timber stock density)",
    "  Demand proxies : 10 operational intensity variables",
    "    (ForestRegen, ForestTending, ConvHighForest, Afforestation,",
    "     Rehabilitation, SoilConservation, BurnedArea, FireSafetyRoad,",
    "     WoodExtraction, RoadRepair)",
    "",
    "H0: all 10 operational coefficients = 0",
    "    (supply alone explains road construction intensity)",
    "",
    SEP,
    "MODEL FIT COMPARISON",
    sep,
    f"  Unrestricted (11 predictors):  logLL = {ll_u:.4f}   AIC = {slm_u.aic:.1f}",
    f"  Restricted   ( 1 predictor):   logLL = {ll_r:.4f}   AIC = {slm_r.aic:.1f}",
    f"  ΔAIC (unrestricted − restricted) = {slm_u.aic - slm_r.aic:.1f}",
    "",
    SEP,
    "LR TEST",
    sep,
    f"  LR statistic = {LR_stat:.2f}",
    f"  df           = {df_lr}",
    f"  p-value      = {p_lr:.6e}  {stars(p_lr)}",
    "",
    SEP,
    "WALD TEST",
    sep,
    f"  Wald chi-sq  = {W_wald:.2f}",
    f"  df           = {len(demand_idx)}",
    f"  p-value      = {p_wald:.6e}  {stars(p_wald)}",
    "",
    SEP,
    "RESTRICTED MODEL (supply only)",
    sep,
]
for i, lbl in enumerate(labels_r):
    lines.append(f"  {lbl:<35s}  {betas_r[i]:>12.6f}   p={p_r[i]:.4f}  {stars(p_r[i])}")

rho_restricted = betas_r[-1]
pseudo_r2_restr = np.corrcoef(slm_r.predy.flatten(), y.flatten())[0,1]**2
pseudo_r2_unres = np.corrcoef(slm_u.predy.flatten(), y.flatten())[0,1]**2

lines += [
    f"  Pseudo R²                          {pseudo_r2_restr:.4f}",
    f"  (Baseline full model Pseudo R²:    {pseudo_r2_unres:.4f})",
    "",
    SEP,
    "INTERPRETATION",
    sep,
    f"  The LR test (LR={LR_stat:.2f}, df=10, p={p_lr:.2e}) strongly rejects H0,",
    f"  confirming that the 10 operational demand proxies collectively explain",
    f"  significant variance in road construction intensity beyond what",
    f"  timber supply (growing stock density) alone can account for.",
    f"",
    f"  The restricted supply-only model achieves Pseudo R²={pseudo_r2_restr:.3f} vs",
    f"  {pseudo_r2_unres:.3f} for the full model — adding operational indicators",
    f"  increases explained variance substantially.",
    f"",
    f"  Growing stock density is non-significant even in the supply-only model",
    f"  (p={p_r[1]:.3f}), indicating that the supply proxy is not a meaningful",
    f"  predictor of road construction intensity by itself.",
    f"",
    f"  Conclusion: road construction intensity in Turkish provinces is driven by",
    f"  operational activity demand, not by standing timber supply levels.",
    SEP,
]

report_path = os.path.join(OUT_DIR, "10_supply_demand_report.txt")
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print(f"\nReport saved: {report_path}")
print("\n" + "\n".join(lines[14:]))
