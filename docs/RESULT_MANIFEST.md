# Result manifest

Paper-facing results are traced to the released source artifacts below.

## Table 1

- Generator: `scripts/reproduce_tables.py`
- Config: `configs/synthetic.yaml`
- Sources:
  - `results/frozen/synthetic/table1_per_scenario.csv` — SHA-256 `84358a7de5a20fc7309cef0da49593af1f75e784d3148a7883c5182d8e38b107`
  - `results/frozen/synthetic/table1_summary.csv` — SHA-256 `60e03af2a3fa852edc48d315dcedf6b464b80ffb7a432eba1da4bfc9dd934c38`
  - `results/frozen/synthetic/table1_paired_bootstrap.csv` — SHA-256 `8045b8beefa041ea9063c356d97bd2b5a3db618019be040f2ca581696757250b`
- Outputs:
  - `results/reproduced/tables/table1_paper.csv` — SHA-256 `0487096d9693b0eda624dbdb105343e6be7b733062b09b9c6c9fea1c563131a8`
  - `results/reproduced/tables/table1_paper.md` — SHA-256 `0452428e89b05f9eaf08cc3150b3b85585281805ddc8e234f376939826bbc8f3`
  - `results/reproduced/tables/table1_paper.tex` — SHA-256 `27eeae069e2b0c690a7d2c9416416686dbd6f7c4d49d77be22b50bc9d7509100`

## Table 2

- Generator: `scripts/reproduce_tables.py`
- Config: `configs/metr_la.yaml`
- Sources:
  - `results/frozen/metr_la/heldout_masks/heldout_10_seed_summary.csv` — SHA-256 `980b817d75a110c0474b56beabead533d88f78172eba3bf023c2ea6c19266bf4`
  - `results/frozen/metr_la/heldout_masks/heldout_10_seed_bootstrap.csv` — SHA-256 `15996e828f97d88f72b364ac33ecd6fa3921b309acec1b00bc66bda9bb02b538`
- Outputs:
  - `results/reproduced/tables/table2_paper.csv` — SHA-256 `f4bd0cf611b5aa5c00d4a0232ab4f1da3f8a8fcafce5483b7dc4313f6a5bb79d`
  - `results/reproduced/tables/table2_paper.md` — SHA-256 `608bd10185ad287fcb655f3ab509b187acbe691a62363471cb6b639bde930fe0`
  - `results/reproduced/tables/table2_paper.tex` — SHA-256 `3520c93424f687ec3620ce080e508cd43b22faedb007dc51e89f2c46dfd43938`

## Fig. 3

- Generator: `scripts/reproduce_figures.py`
- Config: `configs/synthetic.yaml`
- Sources:
  - `results/frozen/synthetic/fig3_bound_data.csv` — SHA-256 `090aea604e9de1494629352c1aad5752b77b48c44886f3e4ab410de948cfef50`
  - `results/frozen/synthetic/fig3_bound_summary.csv` — SHA-256 `f64e8f9d6ffe7e4b4d3fdcd4da7fb2143d1084f91fb513ebe28e0fa271614415`
- Outputs:
  - `figures/fig3_bound_validation.png` — SHA-256 `995e95ae2645d80e5d3a8b3e79eecf4f7a7bb44b32a036d35e444642599a2181`
  - `figures/fig3_bound_validation.pdf` — SHA-256 `8b8a8ee9cef3d42d569489705611a955227201c7079624c25c4d87eab4b7ee17`

## Fig. 4

- Generator: `scripts/reproduce_figures.py`
- Config: `configs/synthetic.yaml`
- Sources:
  - `results/frozen/synthetic/fig4_method_summary.csv` — SHA-256 `b71b253656f5dff03a5e99670832effff494533b74116f4bcb64e47896941ab4`
  - `results/frozen/synthetic/fig4_pairwise_bootstrap.csv` — SHA-256 `7430abda93802de9582315de7655d0e7b9143cf321a045745e1814d0f199a730`
  - `results/frozen/synthetic/fig4_per_graph.csv` — SHA-256 `5f96996b0f968ad3dce0f1587e55d130845746b4699439c1025cadcfd0e934a9`
- Outputs:
  - `figures/fig4_observation_ratio.png` — SHA-256 `aeb9e6941f64cfb4ebd66ef675faecb4da8ec3fbec632ddcc1f66ac1d8a3cf1d`
  - `figures/fig4_observation_ratio.pdf` — SHA-256 `c7647bb9793e3274ab74436caaf263ee08b01e16e0d19e7dbacc4f3b887db400`

## Fig. 5

- Generator: `scripts/reproduce_figures.py`
- Config: `configs/synthetic.yaml`
- Sources:
  - `results/frozen/synthetic/fig5_heatmap_values.csv` — SHA-256 `3647875d914e608655dab9d25e7acfa64117e8985c0069ec3e4624fbdab4a4b8`
- Outputs:
  - `figures/fig5_missingness.png` — SHA-256 `6962a171e39cdbb31735678cb0700be1fb771d1c38432ef93c55fb3db3e7458a`
  - `figures/fig5_missingness.pdf` — SHA-256 `ef3b5a6b3356bb588a859621cc2267437d573ebc4774c47568360f86dd16f380`

## Development ablation

- Verification: `tests/test_frozen_results.py`
- Config: `configs/synthetic.yaml`
- Source:
  - `results/frozen/synthetic/ablation_summary.csv` — SHA-256 `e6af67881b6707330eb8d7a57239d6b97d26724073d4ebec7b028af0bf0f7ea2`

## Seed records

- Synthetic Table 1: `results/frozen/synthetic/table1_seed_manifest.csv`.
- Synthetic robustness: `results/frozen/synthetic/robustness_seed_manifest.csv`.
- METR-LA development mask seeds: `2027101–2027105` (40% and 70% observation per seed).
- METR-LA held-out mask seeds: `2027106–2027115` (40% and 70% observation per seed).
- METR-LA paired bootstrap: 5000 resamples; RNG seeds are stored in `heldout_10_seed_bootstrap.csv`.

## Model artifacts

- ASGC Ridge/StandardScaler: `results/frozen/metr_la/development/asgc_final_model.npz` — SHA-256 `7ee59209f824da8902e7001023dad102d69f720399805adb328b92d14b9857c8`.
- Held-out GAT checkpoints: `results/frozen/metr_la/heldout_masks/models/`.
- Checkpoint hashes: `docs/CHECKPOINT_MANIFEST.json`.
