# -*- coding: utf-8 -*-
"""
01_main_slm.py  —  Primary Spatial Lag Model (Rescaled-X)
----------------------------------------------------------
Replicates the submitted manuscript results exactly:
  rho = 0.444***   Pseudo R² = 0.839   ATI = 1.798

Run from any directory:
  python "D:/Belgelerim/Desktop/Panel Data/Scientific Reports/analysis/01_main_slm.py"

Outputs (in analysis/output/):
  01_slm_coefficients.csv
  01_slm_impacts.csv
  01_slm_results.txt
"""

import os, sys
import numpy as np
import pandas as pd
import scipy.stats
from spreg import ML_Lag
try:
    from statsmodels.stats.diagnostic import het_breuschpagan
    _HAS_SM = True
except ImportError:
    _HAS_SM = False

# ── shared pipeline ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data, X_COLS, Y_COL, OUT_DIR

# ── load data ─────────────────────────────────────────────────────────────────
df, w = load_data()
print(f"Data loaded: {len(df)} provinces, {len(X_COLS)} predictors")

X = df[X_COLS].values
y = df[Y_COL].values.reshape(-1, 1)

# ── fit SLM ───────────────────────────────────────────────────────────────────
print("Fitting SLM ...")
slm = ML_Lag(y, X, w=w, name_y=Y_COL, name_x=X_COLS, vm=True)

betas  = slm.betas.flatten()          # [intercept, b1..bK, rho]
se_all = np.sqrt(np.diag(slm.vm))
z_all  = betas / se_all
p_all  = 2 * (1 - scipy.stats.norm.cdf(np.abs(z_all)))
labels = ["Intercept"] + X_COLS + ["rho"]

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

coef_rows = [
    {"Variable": lbl, "Coef": float(betas[i]), "SE": float(se_all[i]),
     "z": float(z_all[i]), "p": float(p_all[i]), "Sig": stars(p_all[i])}
    for i, lbl in enumerate(labels)
]
coef_df = pd.DataFrame(coef_rows)

# ── Monte Carlo impacts (LeSage & Pace 2009) ─────────────────────────────────
DRAWS, SEED = 20_000, 42
print(f"Computing impacts ({DRAWS:,} MC draws, seed={SEED}) ...")
np.random.seed(SEED)

n      = w.n
W_full = w.full()[0]
I_n    = np.eye(n)
k      = len(X_COLS)

all_draws = np.random.multivariate_normal(betas, slm.vm, size=DRAWS)
dir_arr = np.zeros((DRAWS, k))
ind_arr = np.zeros((DRAWS, k))
tot_arr = np.zeros((DRAWS, k))

for d in range(DRAWS):
    rho_d = float(all_draws[d, -1])
    b_d   = all_draws[d, 1:-1]
    try:
        S = np.linalg.inv(I_n - rho_d * W_full)
    except np.linalg.LinAlgError:
        continue
    tr_S = np.trace(S)
    su_S = float(S.sum())
    for j in range(k):
        dir_arr[d, j] = b_d[j] * tr_S / n
        tot_arr[d, j] = b_d[j] * su_S / n
        ind_arr[d, j] = tot_arr[d, j] - dir_arr[d, j]

imp_rows = []
for j, name in enumerate(X_COLS):
    lo   = float(np.percentile(tot_arr[:, j], 2.5))
    hi   = float(np.percentile(tot_arr[:, j], 97.5))
    p_s  = max(min(2*(tot_arr[:,j]<=0).mean(), 2*(tot_arr[:,j]>=0).mean()), 1/DRAWS)
    imp_rows.append({
        "Variable": name,
        "Direct":   float(dir_arr[:, j].mean()),
        "Indirect": float(ind_arr[:, j].mean()),
        "Total":    float(tot_arr[:, j].mean()),
        "CI_lo": lo, "CI_hi": hi,
        "p_sim": float(p_s), "Sig": stars(p_s),
    })
imp_df = pd.DataFrame(imp_rows)

# ── scalar multipliers ────────────────────────────────────────────────────────
rho_pt = float(slm.rho)
ATI = 1 / (1 - rho_pt)
S_pt = np.linalg.inv(I_n - rho_pt * W_full)   # spatial multiplier at point estimate
ADI = float(np.trace(S_pt)) / n               # n^{-1} tr[(I - rhoW)^{-1}]
AII = ATI - ADI

# ── heteroskedasticity test (Breusch-Pagan, on SLM residuals) ─────────────────
# Figure 5B uses slm.predy (standard fitted values = Xβ + ρWy).
# slm.predy_e (reduced-form) is used only for Spatial Pseudo R².
residuals = slm.u.flatten()
if _HAS_SM:
    X_aug = np.column_stack([np.ones(n), X])
    bp_lm, bp_p, f_stat, f_p = het_breuschpagan(residuals, X_aug)
    het_note = (f"Breusch-Pagan LM = {bp_lm:.4f} (p = {bp_p:.4f}), "
                f"F = {f_stat:.4f} (p = {f_p:.4f})")
else:
    # manual BP: regress residuals^2 on X
    e2 = residuals**2
    X_aug = np.column_stack([np.ones(n), X])
    b_bp = np.linalg.lstsq(X_aug, e2, rcond=None)[0]
    e2_hat = X_aug @ b_bp
    ss_res = ((e2 - e2_hat)**2).sum()
    ss_tot = ((e2 - e2.mean())**2).sum()
    r2_bp = 1 - ss_res/ss_tot
    bp_lm = n * r2_bp
    bp_p = float(1 - scipy.stats.chi2.cdf(bp_lm, df=len(X_COLS)))
    het_note = f"Breusch-Pagan LM = {bp_lm:.4f} (p = {bp_p:.4f}, chi²({len(X_COLS)}))"

# ── save CSVs ─────────────────────────────────────────────────────────────────
coef_df.to_csv(os.path.join(OUT_DIR, "01_slm_coefficients.csv"), index=False, float_format="%.6f")
imp_df.to_csv( os.path.join(OUT_DIR, "01_slm_impacts.csv"),      index=False, float_format="%.6f")

# ── text report ───────────────────────────────────────────────────────────────
SEP = "=" * 70
sep = "-" * 70

lines = [
    SEP,
    "PRIMARY SLM — RESCALED-X  (Revision replication check)",
    f"  Expected:  rho=0.444  Pseudo R²=0.839  ATI=1.798",
    SEP,
    f"  Pseudo R²          : {slm.pr2:.4f}",
    f"  Spatial Pseudo R²  : {slm.pr2_e:.4f}",
    f"  Log-likelihood (LL): {slm.logll:.4f}",
    f"  AIC                : {slm.aic:.4f}",
    f"  BIC (Schwarz)      : {slm.schwarz:.4f}",
    f"  rho                : {rho_pt:.5f}   (p = {p_all[-1]:.6f})  {stars(p_all[-1])}",
    f"  ATI = 1/(1-rho)    : {ATI:.4f}",
    f"  ADI                : {ADI:.4f}",
    f"  AII                : {AII:.4f}",
    "",
    "COEFFICIENTS",
    sep,
    f"{'Variable':42s} {'Coef':>12} {'SE':>10} {'p':>8}  Sig",
    sep,
]
for _, r in coef_df.iterrows():
    lines.append(f"{r['Variable']:42s} {r['Coef']:12.6f} {r['SE']:10.6f} {r['p']:8.4f}  {r['Sig']}")

lines += [
    "",
    "AVERAGE IMPACTS  (20,000 MC draws, seed=42)",
    sep,
    f"{'Variable':42s} {'Direct':>10} {'Indirect':>10} {'Total':>10}  {'95% CI':>24}  p_sim",
    sep,
]
for _, r in imp_df.iterrows():
    ci = f"[{r['CI_lo']:.5f}, {r['CI_hi']:.5f}]"
    lines.append(
        f"{r['Variable']:42s} {r['Direct']:10.5f} {r['Indirect']:10.5f} "
        f"{r['Total']:10.5f}  {ci:>24}  {r['p_sim']:.4f} {r['Sig']}"
    )

lines += [
    "",
    f"Multipliers: ATI={ATI:.4f}  ADI={ADI:.4f}  AII={AII:.4f}",
    f"  ADI = n^{{-1}} tr[(I-rhoW)^{{-1}}]  (corrected from point-estimate multiplier matrix)",
    "",
    "HETEROSKEDASTICITY DIAGNOSTICS (Figure 5B uses slm.predy = Xb + rhoWy)",
    sep,
    f"  {het_note}",
    SEP,
]
report = "\n".join(lines)

out_txt = os.path.join(OUT_DIR, "01_slm_results.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(report)

print("\n" + report)
print(f"\nSaved to: {OUT_DIR}")

# ── quick replication check ───────────────────────────────────────────────────
tol_rho = abs(rho_pt - 0.44381) < 0.001
tol_r2  = abs(slm.pr2 - 0.8388) < 0.001
tol_ati = abs(ATI - 1.7979) < 0.001
print(f"\nReplication check:")
print(f"  rho  match: {'PASS' if tol_rho else 'FAIL'}  (got {rho_pt:.5f}, expected 0.44381)")
print(f"  R²   match: {'PASS' if tol_r2  else 'FAIL'}  (got {slm.pr2:.4f}, expected 0.8388)")
print(f"  ATI  match: {'PASS' if tol_ati else 'FAIL'}  (got {ATI:.4f}, expected 1.7979)")
