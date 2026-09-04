# Forest Road Spatial Lag Model — Türkiye

Replication code for:

> xxxx, x. x. (2026). **Spatially Structured Forest Infrastructure Investment and Provincial Governance Implications in Türkiye.** *xxxxxxxx*. https://doi.org/[ZENODO_DOI]

---

## Overview

This repository contains the analysis code and provincial dataset used to examine spatial patterns in forest road investment across Türkiye's 81 provinces (2017–2024). The study applies Local Indicators of Spatial Association (LISA), Lagrange Multiplier specification tests, and Maximum Likelihood Spatial Lag Modelling (SLM) to GDF annual forestry statistics.

Key findings:
- New road construction intensity is spatially structured across provincial boundaries (ρ = 0.444, p < 0.001; Pseudo R² = 0.839)
- Operational indicators (wood extraction road intensity, road maintenance intensity) are more strongly associated with new construction than standing timber inventory
- Panel Fixed Effects SLM (ρ = 0.628, p < 0.001) is consistent with the cross-sectional result not being driven by time-stable provincial endowments

---

## Repository Structure

```
forest-road-slm-turkiye/
├── data/
│   ├── veri.xlsx            # Provincial forestry statistics, 81 provinces × 8 years (2017–2024)
│   └── turkiye_il.*         # Province boundary shapefile (OpenStreetMap contributors, ODbL)
├── 01_main_slm.py           # ESDA, LISA, LM tests, SLM estimation, impacts
├── 03_vif_standardized.py   # VIF table and standardized beta-weights
├── 04_panel_resm.py         # Panel FE-SLM, RE-SLM, Hausman test
├── 05_regime_interaction.py # LISA cluster map, regime interaction LR test
├── 06_robustness.py         # Alternative weights matrices, terrain robustness
├── 07_sdm_lr_test.py        # Formal SDM vs. SLM likelihood-ratio test
├── data_loader.py           # Shared data loading utilities
├── requirements.txt
└── README.md
```

---

## Data

`data/veri.xlsx` — 648 rows (81 provinces × 8 years), 23 columns. Variables derived from General Directorate of Forestry (GDF) annual statistical tables. Column names are in English. See Table 1 of the manuscript for full variable descriptions.

| Column | Description | Unit |
|---|---|---|
| PROV_ID | Province code | — |
| Province | Province name | — |
| Year | Year | 2017–2024 |
| Forest_Area_ha | Provincial forest area | ha |
| GrowingStock_m3 | Growing stock volume | m³ |
| NewRoad_km | New road construction | km |
| RoadMajorRepair_km | Major road repair | km |
| FireSafetyRoad_km | Fire safety road construction | km |
| WoodExtraction_* | Wood extraction road intensity | km |
| ForestRegen_ha | Forest regeneration activity | ha |
| Afforestation_ha | Afforestation activity | ha |
| ... | (see manuscript Table 1) | — |

`data/turkiye_il.*` — Province boundary shapefile. Source: OpenStreetMap contributors (ODbL). Projected to EPSG:5637 for spatial analysis.

---

## Requirements

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
```

Core dependencies: `geopandas`, `spreg`, `libpysal`, `esda`, `matplotlib`, `pandas`, `numpy`, `scipy`, `openpyxl`.

---

## How to Run

Run scripts in numbered order. Each script reads from `data/` and writes to `output/`.

```bash
python 01_main_slm.py        # Baseline SLM, LISA, LM tests, impacts table
python 03_vif_standardized.py  # VIF table, standardized betas
python 04_panel_resm.py      # Panel FE/RE SLM, Hausman test
python 05_regime_interaction.py  # LISA cluster maps, interaction LR test
python 06_robustness.py      # Weights matrix robustness, terrain control
python 07_sdm_lr_test.py     # Formal SDM vs. SLM LR test
```

All outputs are saved to `output/` as `.txt` files. Runtime: approximately 5–10 minutes total on a standard desktop (the panel script is slowest).

---

## Spatial Setup

The Queen-contiguity spatial weights matrix is constructed from `turkiye_il.shp` using `libpysal.weights.Queen`. Row-standardization is applied before all spatial regression steps. The projected CRS used for planar distance calculations is EPSG:5637 (Turkish National Grid).

---

## License

Code: MIT License.
Data (`veri.xlsx`): Derived from publicly available GDF annual statistics (https://www.ogm.gov.tr). Free to use with attribution.
Province boundaries: OpenStreetMap contributors, ODbL (https://www.openstreetmap.org/copyright).

---

## Citation

If you use this code or data, please cite the paper and the repository:

```bibtex
@article{hatay2026,
  author  = {xxxxx, xxxx},
  title   = {Spatially Structured Forest Infrastructure Investment and Provincial Governance Implications in Türkiye},
  journal = {xxxxxxxx},
  year    = {2026},
  doi     = {[ZENODO_DOI]}
}
```

---

## Contact

xxxxxxxxxxxxx (PhD). — Department of xxxxxxxxxx, xxxxxxxxxxxxxxx
xxxxx@xxx.edu.tr
