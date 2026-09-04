# -*- coding: utf-8 -*-
"""
data_loader.py  —  Shared data loading pipeline
------------------------------------------------
Reads data/veri.xlsx + data/iliski.swm, computes per-hectare intensities,
applies rescaling (×1000) and log1p transformation, returns province means
aligned to the spatial weights matrix.

All analysis scripts import from here for a single consistent pipeline.

Variable naming convention:
  Raw intensities : <var>_kmha  (km per ha) or <var>_haha  (ha per ha)
  Model inputs    : <var>_log   (log1p of rescaled intensity)
"""

import os
import numpy as np
import pandas as pd
import libpysal
from libpysal.weights import WSP

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))   # Scientific Reports/data/
DATA_XLS = os.path.join(DATA_DIR, "veri.xlsx")
DATA_SWM = os.path.join(DATA_DIR, "iliski.swm")

# Output directory
OUT_DIR = os.path.join(_HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Model variable names ───────────────────────────────────────────────────────
# Dependent variable
Y_COL = "NewRoad_kmha"

# Independent variables (log1p of rescaled intensities)
X_COLS = [
    "GrowingStock_m3ha_log",      # Growing stock density       (m³/ha, log1p)
    "ForestRegen_haha_log",        # Forest regeneration         (ha/ha ×1000, log1p)
    "ForestTending_haha_log",      # Forest tending              (ha/ha ×1000, log1p)
    "ConvHighForest_haha_log",     # Conversion to high forest   (ha/ha ×1000, log1p)
    "Afforestation_haha_log",      # Afforestation               (ha/ha ×1000, log1p)
    "Rehabilitation_haha_log",     # Rehabilitation               (ha/ha ×1000, log1p)
    "SoilConservation_haha_log",   # Soil conservation           (ha/ha ×1000, log1p)
    "BurnedArea_haha_log",         # Burned area                 (ha/ha ×1000, log1p)
    "FireSafetyRoad_kmha_log",     # Fire safety road intensity  (km/ha ×1000, log1p)
    "WoodExtraction_kmha_log",     # Wood extraction road intens.(km/ha ×1000, log1p)
    "RoadRepair_kmha_log",         # Road repair intensity       (km/ha ×1000, log1p)
]

# Human-readable labels (used in reports and plots)
X_LABELS = {
    "GrowingStock_m3ha_log":    "Growing stock density",
    "ForestRegen_haha_log":     "Forest regeneration",
    "ForestTending_haha_log":   "Forest tending",
    "ConvHighForest_haha_log":  "Conv. to high forest",
    "Afforestation_haha_log":   "Afforestation",
    "Rehabilitation_haha_log":  "Rehabilitation",
    "SoilConservation_haha_log":"Soil conservation",
    "BurnedArea_haha_log":      "Burned area",
    "FireSafetyRoad_kmha_log":  "Fire safety roads",
    "WoodExtraction_kmha_log":  "Wood extraction roads",
    "RoadRepair_kmha_log":      "Road repair intensity",
}


# ── Spatial weights ────────────────────────────────────────────────────────────
def load_weights():
    swm = libpysal.io.open(DATA_SWM, mode="r")
    wsp = swm.read()
    w = WSP.to_W(wsp)
    w.transform = "r"
    return w


def align_to_weights(df, w, id_col="PROV_ID"):
    return df.set_index(id_col).reindex(w.id_order).reset_index()


# ── Main loader ───────────────────────────────────────────────────────────────
def load_data():
    """
    Returns (df_mean, w):
      df_mean — 81-row province-mean DataFrame with model-ready columns
      w       — row-standardised queen contiguity spatial weights
    """
    df = pd.read_excel(DATA_XLS)

    # Fill structural NAs in road maintenance columns
    for c in ["ArtStructure_km", "RoadMajorRepair_km", "RoadSurface_km"]:
        df[c] = df[c].fillna(0)

    # ── Composite variables ────────────────────────────────────────────────────
    # Fire safety road = FireSafetyRoad + TowerCabinRoad
    df["FireSafetyRoad_composite_km"] = df["FireSafetyRoad_km"] + df["TowerCabinRoad_km"]
    # Wood extraction road = TractorRoad + DepotRoad
    df["WoodExtraction_km"]           = df["TractorRoad_km"]  + df["DepotRoad_km"]
    # Road repair = MajorRepair + Surface construction
    df["RoadRepair_km"]               = df["RoadMajorRepair_km"] + df["RoadSurface_km"]

    # ── Per-hectare intensities ────────────────────────────────────────────────
    A = df["Forest_Area_ha"]   # denominator

    df["GrowingStock_m3ha"]    = df["GrowingStock_m3"]               / A
    df["NewRoad_kmha"]         = df["NewRoad_km"]                    / A   # Y
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

    # ── Province means (2017–2024) ─────────────────────────────────────────────
    mean_cols = ["PROV_ID",
                 "GrowingStock_m3ha", "NewRoad_kmha",
                 "ForestRegen_haha", "ForestTending_haha", "ConvHighForest_haha",
                 "Afforestation_haha", "Rehabilitation_haha", "SoilConservation_haha",
                 "BurnedArea_haha", "FireSafetyRoad_kmha", "WoodExtraction_kmha",
                 "RoadRepair_kmha"]
    df_mean = df[mean_cols].groupby("PROV_ID").mean().reset_index()

    # ── Rescale ×1000 then log1p ───────────────────────────────────────────────
    # ha/ha variables: ×1000 so unit = "ha per 1 000 ha of forest"
    HA_HA_COLS = ["ForestRegen_haha", "ForestTending_haha", "ConvHighForest_haha",
                  "Afforestation_haha", "Rehabilitation_haha", "SoilConservation_haha",
                  "BurnedArea_haha"]
    # km/ha variables: ×1000 so unit = "m per ha" (i.e. metres per ha)
    KM_HA_COLS = ["FireSafetyRoad_kmha", "WoodExtraction_kmha", "RoadRepair_kmha"]

    for col in HA_HA_COLS + KM_HA_COLS:
        df_mean[col + "_log"] = np.log1p(df_mean[col] * 1000)

    # Growing stock: m³/ha — no ×1000 rescaling needed (already in meaningful units)
    df_mean["GrowingStock_m3ha_log"] = np.log1p(df_mean["GrowingStock_m3ha"])

    # ── Align to weights matrix order ──────────────────────────────────────────
    w = load_weights()
    df_mean["PROV_ID"] = df_mean["PROV_ID"].astype(int)
    df_mean = align_to_weights(df_mean, w)

    return df_mean, w
