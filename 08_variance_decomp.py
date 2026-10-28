# -*- coding: utf-8 -*-
"""
08_variance_decomp.py  —  Between/Within variance decomposition for Table S1
----------------------------------------------------------------------------
Computes ANOVA-style between-province / within-province variance decomposition
on the full panel (81 provinces × 8 years = 648 obs).

Outputs:
  output/08_variance_decomp.csv
  output/08_variance_decomp.txt
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))
DATA_XLS = os.path.join(DATA_DIR, "veri.xlsx")
OUT_DIR  = os.path.join(_HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── load full panel ────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_XLS)

for c in ["ArtStructure_km", "RoadMajorRepair_km", "RoadSurface_km"]:
    df[c] = df[c].fillna(0)

df["FireSafetyRoad_composite_km"] = df["FireSafetyRoad_km"] + df["TowerCabinRoad_km"]
df["WoodExtraction_km"]           = df["TractorRoad_km"]  + df["DepotRoad_km"]
df["RoadRepair_km"]               = df["RoadMajorRepair_km"] + df["RoadSurface_km"]

A = df["Forest_Area_ha"]
df["GrowingStock_m3ha"]    = df["GrowingStock_m3"]               / A
df["NewRoad_kmha"]         = df["NewRoad_km"]                    / A
df["FireSafetyRoad_kmha"]  = df["FireSafetyRoad_composite_km"]  / A
df["WoodExtraction_kmha"]  = df["WoodExtraction_km"]             / A
df["RoadRepair_kmha"]      = df["RoadRepair_km"]                 / A
df["ForestRegen_haha"]     = df["ForestRegen_ha"]               / A
df["ForestTending_haha"]   = df["ForestTending_ha"]             / A
df["ConvHighForest_haha"]  = df["ConvHighForest_ha"]            / A
df["Afforestation_haha"]   = df["Afforestation_ha"]             / A
df["Rehabilitation_haha"]  = df["Rehabilitation_ha"]            / A
df["SoilConservation_haha"]= df["SoilConservation_ha"]          / A
df["BurnedArea_haha"]      = df["BurnedArea_ha"]                / A
df = df.fillna(0)

RAW_VARS = {
    "Growing stock density":    ("GrowingStock_m3ha",    "m³/ha"),
    "New road construction":    ("NewRoad_kmha",         "km/ha"),
    "Forest regeneration":      ("ForestRegen_haha",     "ha/ha"),
    "Forest tending":           ("ForestTending_haha",   "ha/ha"),
    "Conv. to high forest":     ("ConvHighForest_haha",  "ha/ha"),
    "Afforestation intensity":  ("Afforestation_haha",   "ha/ha"),
    "Rehabilitation intensity": ("Rehabilitation_haha",  "ha/ha"),
    "Soil conservation":        ("SoilConservation_haha","ha/ha"),
    "Burned area":              ("BurnedArea_haha",      "ha/ha"),
    "Fire response roads":      ("FireSafetyRoad_kmha",  "km/ha"),
    "Wood extraction roads":    ("WoodExtraction_kmha",  "km/ha"),
    "Road maintenance":         ("RoadRepair_kmha",      "km/ha"),
}

# Log-transformed model variables
HA_HA_COLS = ["ForestRegen_haha", "ForestTending_haha", "ConvHighForest_haha",
              "Afforestation_haha", "Rehabilitation_haha", "SoilConservation_haha",
              "BurnedArea_haha"]
KM_HA_COLS = ["FireSafetyRoad_kmha", "WoodExtraction_kmha", "RoadRepair_kmha"]

for col in HA_HA_COLS + KM_HA_COLS:
    df[col + "_log"] = np.log1p(df[col] * 1000)
df["GrowingStock_m3ha_log"] = np.log1p(df["GrowingStock_m3ha"])
df["NewRoad_kmha_log"]      = np.log1p(df["NewRoad_kmha"] * 1000)  # for reference only

LOG_VARS = {
    "Growing stock density":    ("GrowingStock_m3ha_log",    "log1p(m³/ha)"),
    "New road construction":    ("NewRoad_kmha",              "km/ha (raw Y)"),
    "Forest regeneration":      ("ForestRegen_haha_log",      "log1p(×1000 ha/ha)"),
    "Forest tending":           ("ForestTending_haha_log",    "log1p(×1000 ha/ha)"),
    "Conv. to high forest":     ("ConvHighForest_haha_log",   "log1p(×1000 ha/ha)"),
    "Afforestation intensity":  ("Afforestation_haha_log",    "log1p(×1000 ha/ha)"),
    "Rehabilitation intensity": ("Rehabilitation_haha_log",   "log1p(×1000 ha/ha)"),
    "Soil conservation":        ("SoilConservation_haha_log", "log1p(×1000 ha/ha)"),
    "Burned area":              ("BurnedArea_haha_log",       "log1p(×1000 ha/ha)"),
    "Fire response roads":      ("FireSafetyRoad_kmha_log",   "log1p(×1000 km/ha)"),
    "Wood extraction roads":    ("WoodExtraction_kmha_log",   "log1p(×1000 km/ha)"),
    "Road maintenance":         ("RoadRepair_kmha_log",       "log1p(×1000 km/ha)"),
}


def decompose(df_panel, varname, id_col="PROV_ID"):
    x = df_panel[varname].values
    ids = df_panel[id_col].values
    grand_mean = x.mean()
    prov_ids = np.unique(ids)
    N = len(prov_ids)
    T_per = np.array([np.sum(ids == p) for p in prov_ids])
    prov_means = np.array([x[ids == p].mean() for p in prov_ids])
    T_common = int(np.median(T_per))

    # ANOVA sums of squares
    ss_between = np.sum(T_per * (prov_means - grand_mean)**2)
    ss_within  = np.sum((x - prov_means[np.searchsorted(prov_ids, ids)])**2)
    ss_total   = ss_between + ss_within

    df_between = N - 1
    df_within  = len(x) - N

    ms_between = ss_between / df_between
    ms_within  = ss_within  / df_within

    bw_ratio      = ms_between / ms_within if ms_within > 0 else np.inf
    between_share = 100 * ss_between / ss_total if ss_total > 0 else np.nan

    return {
        "SS_between": ss_between,
        "SS_within":  ss_within,
        "SS_total":   ss_total,
        "MS_between": ms_between,
        "MS_within":  ms_within,
        "BW_ratio":   bw_ratio,
        "Between_share_pct": between_share,
        "N_prov": N,
        "N_obs":  len(x),
    }


SEP = "=" * 80
sep = "-" * 80

lines = [SEP, "BETWEEN / WITHIN VARIANCE DECOMPOSITION  (panel: N=81, T=8, n=648)",
         "Compare with Table S1 in Supplementary_revision.md", SEP]

rows_raw = []
rows_log = []

lines.append("\n--- RAW (untransformed) variables ---")
lines.append(f"{'Variable':35s}  {'B/W ratio':>10}  {'Between%':>10}  {'match S1?':>10}")
lines.append(sep)

TABLE_S1 = {
    "Growing stock density":    (829.8, 99.2),
    "Conv. to high forest":     (72.4,  91.1),
    "Forest tending":           (22.0,  75.6),
    "Fire response roads":      (9.4,   57.1),
    "New road construction":    (6.9,   49.2),
    "Road maintenance":         (5.4,   43.4),
    "Forest regeneration":      (5.4,   43.2),
    "Soil conservation":        (4.6,   39.3),
    "Afforestation intensity":  (4.4,   38.1),
    "Wood extraction roads":    (3.1,   30.6),
    "Rehabilitation intensity": (2.9,   29.2),
    "Burned area":              (1.5,   17.1),
}

for label, (col, unit) in RAW_VARS.items():
    r = decompose(df, col)
    s1_ratio, s1_share = TABLE_S1.get(label, (None, None))
    match = ""
    if s1_ratio is not None:
        diff_r = abs(r["BW_ratio"] - s1_ratio)
        diff_s = abs(r["Between_share_pct"] - s1_share)
        match = "OK" if diff_r < 0.5 and diff_s < 1 else f"DIFF(ratio={diff_r:.1f}, share={diff_s:.1f})"
    lines.append(f"{label:35s}  {r['BW_ratio']:10.1f}  {r['Between_share_pct']:10.1f}  {match}")
    rows_raw.append({"Variable": label, "Unit": unit,
                     "BW_ratio_raw": round(r["BW_ratio"],1),
                     "Between_pct_raw": round(r["Between_share_pct"],1),
                     "S1_BW": s1_ratio, "S1_share": s1_share})

lines.append("")
lines.append("--- LOG-TRANSFORMED model variables ---")
lines.append(f"{'Variable':35s}  {'B/W ratio':>10}  {'Between%':>10}  {'match S1?':>10}")
lines.append(sep)

for label, (col, unit) in LOG_VARS.items():
    r = decompose(df, col)
    s1_ratio, s1_share = TABLE_S1.get(label, (None, None))
    match = ""
    if s1_ratio is not None:
        diff_r = abs(r["BW_ratio"] - s1_ratio)
        diff_s = abs(r["Between_share_pct"] - s1_share)
        match = "OK" if diff_r < 0.5 and diff_s < 1 else f"DIFF(ratio={diff_r:.1f}, share={diff_s:.1f})"
    lines.append(f"{label:35s}  {r['BW_ratio']:10.1f}  {r['Between_share_pct']:10.1f}  {match}")
    rows_log.append({"Variable": label, "Unit": unit,
                     "BW_ratio_log": round(r["BW_ratio"],1),
                     "Between_pct_log": round(r["Between_share_pct"],1),
                     "S1_BW": s1_ratio, "S1_share": s1_share})

lines.append("")
lines.append("--- SUMMARY ---")
n_below_50_raw = sum(1 for r in rows_raw if r["Between_pct_raw"] < 50)
n_below_50_log = sum(1 for r in rows_log if r["Between_pct_log"] < 50)
lines.append(f"Raw vars below 50% between-share: {n_below_50_raw}/12")
lines.append(f"Log vars below 50% between-share: {n_below_50_log}/12")
lines.append(SEP)

report = "\n".join(lines)
print(report)

# ── save ──────────────────────────────────────────────────────────────────────
out_txt = os.path.join(OUT_DIR, "08_variance_decomp.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(report)

df_raw = pd.DataFrame(rows_raw)
df_log = pd.DataFrame(rows_log)
df_all = df_raw.merge(df_log[["Variable","BW_ratio_log","Between_pct_log"]], on="Variable")
df_all.to_csv(os.path.join(OUT_DIR, "08_variance_decomp.csv"), index=False, float_format="%.2f")

print(f"\nSaved to: {OUT_DIR}")
