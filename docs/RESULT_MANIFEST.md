# Result manifest

Paper-facing results are traced to the released source artifacts below.

## Table 1 - synthetic graphs

- Generator: `scripts/reproduce_tables.py`
- Config: `configs/synthetic.yaml`
- Sources:
  - `results/frozen/synthetic/table1_per_scenario.csv` - SHA-256 `84358a7de5a20fc7309cef0da49593af1f75e784d3148a7883c5182d8e38b107`
  - `results/frozen/synthetic/table1_summary.csv` - SHA-256 `60e03af2a3fa852edc48d315dcedf6b464b80ffb7a432eba1da4bfc9dd934c38`
  - `results/frozen/synthetic/table1_paired_bootstrap.csv` - SHA-256 `8045b8beefa041ea9063c356d97bd2b5a3db618019be040f2ca581696757250b`
- Outputs:
  - `results/reproduced/tables/table1_paper.csv` - SHA-256 `0487096d9693b0eda624dbdb105343e6be7b733062b09b9c6c9fea1c563131a8`
  - `results/reproduced/tables/table1_paper.md` - SHA-256 `0452428e89b05f9eaf08cc3150b3b85585281805ddc8e234f376939826bbc8f3`
  - `results/reproduced/tables/table1_paper.tex` - SHA-256 `27eeae069e2b0c690a7d2c9416416686dbd6f7c4d49d77be22b50bc9d7509100`

## Table 2 - node-disjoint METR-LA

- Generator: `scripts/reproduce_tables.py`
- Experiment entry point: `scripts/run_metr_la_topology_disjoint.py`
- Config: `configs/metr_la.yaml`
- Sources:
  - `results/frozen/metr_la_topology_disjoint/test_summary.csv` - SHA-256 `13327461a8893576e9bc31ecdc6639bf31874514b3636dcb1c9e7948e4654971`
  - `results/frozen/metr_la_topology_disjoint/graph_exact_signflip.csv` - SHA-256 `1a1920e04a537938fa47864ca1e261f92c8020ca7ec82ed5ba3738695d4eedcf`
  - `results/frozen/metr_la_topology_disjoint/test_scenario_metrics.csv` - SHA-256 `e290b5c618fed84e6e669062f7075ed5629f8d39f27b2a57b8cae4360488d957`
- Outputs:
  - `results/reproduced/tables/table2_paper.csv` - SHA-256 `80b2b51eac03cc486ba5e850ac968d2f0c7b26db1ce324f36d55d28e51f2cdaa`
  - `results/reproduced/tables/table2_paper.md` - SHA-256 `1291244d70da8be133fa8ac046995c83f4d347e9c0c851cd06f7c7043dd87c02`
  - `results/reproduced/tables/table2_paper.tex` - SHA-256 `2d497c4b58019adf1ab9fe9597c2fff1bc65fee95330ee8f103de9253c138ea0`

## METR-LA protocol/statistics artifacts

- `results/frozen/metr_la_topology_disjoint/node_split.csv` - SHA-256 `e3d2b8acd4b87145330e784f016b5890a1b1cd6e3f82abeab8191941504d531b`
- `results/frozen/metr_la_topology_disjoint/protocol_manifest.json` - SHA-256 `02cf93a55a1f3a9b7d759efd9326ebc0e8f501dfbd32a5275c35c561fd5a46d0`
- `results/frozen/metr_la_topology_disjoint/lambda_graph_cv.csv` - SHA-256 `d27e613267870058d4265e7e7ae7b5586ddfd1fa6026ab5af0cecfdc6186e1bc`
- `results/frozen/metr_la_topology_disjoint/development_calibrator.npz` - SHA-256 `b1be17f82e2e2a9e0366e6c039a1c6aa18f6785952f2cc4f239fb9c2d6d43269`
- `results/frozen/metr_la_topology_disjoint/development_gat_training.csv` - SHA-256 `9e5f0aff47e3cf40dddcbec4131e932eb857e15a5686042ad9131f89929be51c`
- `results/frozen/metr_la_topology_disjoint/test_gat_training.csv` - SHA-256 `1846620b6ebe06f9cbb30166727bb17473162fe5b96e0eae03f10fd8684d31f1`
- `results/frozen/metr_la_topology_disjoint/graph_cluster_bootstrap.csv` - SHA-256 `33a1e292a70039b68b6a780cb32c01e8447ac46d06821c767ae84c2a5737c9a6`
- `results/frozen/metr_la_topology_disjoint/graph_exact_signflip.csv` - SHA-256 `1a1920e04a537938fa47864ca1e261f92c8020ca7ec82ed5ba3738695d4eedcf`
- `results/frozen/metr_la_topology_disjoint/asgc_relative_reductions.csv` - SHA-256 `37d44883b4c9388734d113b219fab1874d4368b686854a55c02b5530057300bd`
- `results/frozen/metr_la_topology_disjoint/validation_audit.json` - SHA-256 `89654f9dd036a826592f6d9fbbb7d47ff92ed4b634d6c1f289afa2036d1e44bc`
- `results/frozen/metr_la_topology_disjoint/SHA256SUMS.txt` - SHA-256 `6a5bd00f7ee5a60fab79c6862447d0171b36dd5cf39111bea99d7fe33a2781a5`

The selected Ridge regularization is `lambda = 100`, chosen by leave-one-development-graph-out cross-validation. The six test subsets are node-disjoint and each has two mask repetitions at 40% and 70% observation.

## Fig. 3

- Generator: `scripts/reproduce_figures.py`
- Sources:
  - `results/frozen/synthetic/fig3_bound_data.csv` - SHA-256 `090aea604e9de1494629352c1aad5752b77b48c44886f3e4ab410de948cfef50`
  - `results/frozen/synthetic/fig3_bound_summary.csv` - SHA-256 `f64e8f9d6ffe7e4b4d3fdcd4da7fb2143d1084f91fb513ebe28e0fa271614415`
- Outputs:
  - `figures/fig3_bound_validation.png` - SHA-256 `995e95ae2645d80e5d3a8b3e79eecf4f7a7bb44b32a036d35e444642599a2181`
  - `figures/fig3_bound_validation.pdf` - SHA-256 `8b8a8ee9cef3d42d569489705611a955227201c7079624c25c4d87eab4b7ee17`

## Fig. 4

- Generator: `scripts/reproduce_figures.py`
- Sources:
  - `results/frozen/synthetic/fig4_method_summary.csv` - SHA-256 `b71b253656f5dff03a5e99670832effff494533b74116f4bcb64e47896941ab4`
  - `results/frozen/synthetic/fig4_pairwise_bootstrap.csv` - SHA-256 `7430abda93802de9582315de7655d0e7b9143cf321a045745e1814d0f199a730`
  - `results/frozen/synthetic/fig4_per_graph.csv` - SHA-256 `5f96996b0f968ad3dce0f1587e55d130845746b4699439c1025cadcfd0e934a9`
- Outputs:
  - `figures/fig4_observation_ratio.png` - SHA-256 `aeb9e6941f64cfb4ebd66ef675faecb4da8ec3fbec632ddcc1f66ac1d8a3cf1d`
  - `figures/fig4_observation_ratio.pdf` - SHA-256 `c7647bb9793e3274ab74436caaf263ee08b01e16e0d19e7dbacc4f3b887db400`

## Fig. 5

- Generator: `scripts/reproduce_figures.py`
- Source:
  - `results/frozen/synthetic/fig5_heatmap_values.csv` - SHA-256 `3647875d914e608655dab9d25e7acfa64117e8985c0069ec3e4624fbdab4a4b8`
- Outputs:
  - `figures/fig5_missingness.png` - SHA-256 `6962a171e39cdbb31735678cb0700be1fb771d1c38432ef93c55fb3db3e7458a`
  - `figures/fig5_missingness.pdf` - SHA-256 `ef3b5a6b3356bb588a859621cc2267437d573ebc4774c47568360f86dd16f380`

## Development ablation

- Source: `results/frozen/synthetic/ablation_summary.csv` - SHA-256 `e6af67881b6707330eb8d7a57239d6b97d26724073d4ebec7b028af0bf0f7ea2`
- Verification: `tests/test_frozen_results.py`
