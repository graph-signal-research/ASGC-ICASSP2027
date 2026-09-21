# Aggregation-Aware Strength-Guided Calibration for Information Fusion over Incomplete Directed Graphs

Code and reproducibility artifacts for the ICASSP 2027 submission:

> Anonymous authors, “Aggregation-Aware Strength-Guided Calibration for Information Fusion over Incomplete Directed Graphs,” under review, 2027.

## Overview

Aggregation-Aware Strength-Guided Calibration (ASGC) studies directed graph completion from the perspective of normalized information fusion. A GAT first estimates unobserved directed relation weights. ASGC then calibrates target-node in-strengths used by normalization while leaving the preliminary GAT edge estimates unchanged.

```text
partial directed graph
        ↓
GAT preliminary edge completion
        ↓
target-wise in-strength calibration
        ↓
normalized aggregation weights
        ↓
fused signal
```

For target node `i`, the calibration feature vector is

```text
[tilde_c_i, o_i, p_i, n_i^o, n_i^m, mu_i^o, mu_i^m, sigma_i^m]
```

Ridge regression estimates the in-strength residual. The calibrated value is projected to `[o_i, o_i + n_i^m]` and normalized into aggregation weights. Raw-GAT and ASGC use the same preliminary completed adjacency, while ASGC calibrates the downstream in-strength representation.

## Repository Structure

```text
configs/                  experiment configurations
src/
  data/                   METR-LA loading and matrix convention
  models/                 GAT and ASGC implementations
  metrics.py              evaluation metrics
  utils.py                configuration and hashing utilities
scripts/                  experiment and reproduction entry points
tests/                    automated consistency tests
results/
  frozen/
    synthetic/            paper synthetic result artifacts
    metr_la_topology_disjoint/  paper METR-LA result artifacts
  reproduced/             paper tables and summaries generated from stored results
figures/                  method overview and synthetic-result figures
docs/                     data, protocol, reproducibility, and provenance
data/METR-LA/             local location for user-prepared METR-LA files
```

## Paper-to-Code Map

| Paper component | Implementation |
|---|---|
| Method overview figure | `figures/fig2.pdf` |
| GAT preliminary edge completion | `src/models/gat.py` |
| ASGC in-strength calibration | `src/models/asgc.py` |
| METR-LA loading / graph convention | `src/data/metr_la.py` |
| Node-disjoint METR-LA experiment | `scripts/run_metr_la_topology_disjoint.py` |
| Evaluation metrics | `src/metrics.py` |
| Table 1–2 reproduction | `scripts/reproduce_tables.py` |
| Fig. 3–5 reproduction | `scripts/reproduce_figures.py` |

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate asgc
```

### pip / venv

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows
python -m pip install -r requirements.txt
```

## Data

### Synthetic experiments

Synthetic scenario-level records, bootstrap summaries, bound data, observation-ratio results, missingness results, and ablation values are stored under `results/frozen/synthetic/`.

### METR-LA

Raw METR-LA measurements are not redistributed. Prepare the public DCRNN/METR-LA files under `data/METR-LA/`:

```bash
python scripts/download_metr_la.py --config configs/metr_la.yaml
python scripts/download_metr_la.py --config configs/metr_la.yaml \
    --metr-h5 /path/to/metr-la.h5
```

Detailed data instructions and reference hashes are in [`docs/DATA.md`](docs/DATA.md).

The directed-matrix convention is

```text
A[j, i] = j -> i
c = A.T @ ones(N)
```

Self-loops are removed and the official adjacency matrix is not transposed.

For the paper METR-LA experiment, ten mutually node-disjoint 20-sensor subsets are fixed from sensor-location metadata before the adjacency matrix is loaded. Four subsets are used for development and six for testing. Every subset uses all `20 x 19 = 380` directed non-self pairs as candidate relations, independently of the reference adjacency.

## Quick Start

```bash
python scripts/run_synthetic.py --config configs/synthetic.yaml --smoke-test
python scripts/reproduce_tables.py --smoke-test
python scripts/reproduce_figures.py --config configs/synthetic.yaml --smoke-test
pytest -q
```

Smoke tests write only to temporary directories and leave the checked-in
paper tables and figures unchanged.

A full METR-LA rerun requires the prepared raw dataset:

```bash
python scripts/run_metr_la_topology_disjoint.py
```

## Reproducing Paper Results

### Synthetic summary and figures

```bash
python scripts/run_synthetic.py --config configs/synthetic.yaml
python scripts/reproduce_figures.py --config configs/synthetic.yaml
```

The synthetic script summarizes the stored per-scenario results and checks that Raw-GAT and ASGC have identical missing-edge MAE in every scenario. Fig. 3–5 are generated from the accompanying synthetic result files.

Figure values and physical layouts are reproducible from the released data.
Binary PDF/PNG hashes can vary across systems when the configured serif fonts
resolve to different installed font files or rendering backends.

### Node-disjoint METR-LA

```bash
python scripts/run_metr_la_topology_disjoint.py
```

The experiment:

- fixes four development and six test 20-sensor subsets using only `graph_sensor_locations.csv` and deterministic Morton/Z-order grouping;
- keeps all ten subsets mutually node-disjoint;
- uses all 380 directed non-self candidate pairs per subset;
- uses two deterministic mask repetitions at 40% and 70% observation;
- trains a fresh graph-specific GAT for each graph/mask/ratio, with fit-observed
  candidates in the training loss and validation-observed candidates in the
  validation loss; only observed positive relations are used for message passing;
- chooses Ridge regularization by leave-one-development-graph-out validation;
- fits the StandardScaler and Ridge calibrator exclusively on development graphs;
- freezes the calibrator before evaluating the six test subsets;
- uses the first 70% of the traffic series for node descriptors and the following 10% only for temporal fusion evaluation.

The METR-LA results reported in the paper are stored under `results/frozen/metr_la_topology_disjoint/`. Full details are in [`docs/TOPOLOGY_DISJOINT_METRLA.md`](docs/TOPOLOGY_DISJOINT_METRLA.md).

### Table 1 and Table 2

```bash
python scripts/reproduce_tables.py \
    --config configs/synthetic.yaml \
    --metr-config configs/metr_la.yaml
```

Generated files:

```text
results/reproduced/tables/table1_paper.csv
results/reproduced/tables/table1_paper.md
results/reproduced/tables/table1_paper.tex
results/reproduced/tables/table2_paper.csv
results/reproduced/tables/table2_paper.md
results/reproduced/tables/table2_paper.tex
```

## Metrics

- `E_a_all`: MAE over all hidden candidate pairs in the METR-LA experiment.
- `E_a_positive`: MAE over hidden candidate pairs whose reference weight is positive; reference positivity is used only for evaluation.
- `E_c`: graph-level L1 error of target-node in-strengths.
- `E_w`: L1 error between normalized aggregation-weight vectors.
- `E_x`: channel-weighted L1 fused-signal error used for synthetic experiments.
- `E_{x,2}^{METR}`: mean temporal L2 fused-signal deviation used for METR-LA.

Lower values indicate better performance for all reported error metrics.

## Key Findings

### Synthetic graphs

| Metric | Raw-GAT | ASGC | Relative reduction |
|---|---:|---:|---:|
| `E_c` | 3.7426 | **3.0947** | **17.3%** |
| `E_w` | 0.0777 | **0.0668** | **14.0%** |
| `E_x` | 0.0050 | **0.0043** | **14.4%** |

Raw-GAT and ASGC have identical preliminary edge completion. NodeMean obtains lower mean edge MAE than Raw-GAT while producing worse downstream fusion errors, illustrating that edge-level recovery accuracy alone does not determine aggregation accuracy.

### Node-disjoint METR-LA

| Observation | Metric | Raw-GAT | ASGC | Relative reduction |
|---:|---|---:|---:|---:|
| 40% | `E_c` | 20.0814 | **14.8649** | **25.98%** |
| 40% | `E_w` | 0.5321 | **0.3821** | **28.20%** |
| 40% | `E_{x,2}^{METR}` | 4.8785 | **2.9051** | **40.45%** |
| 70% | `E_c` | 10.7365 | **9.8704** | **8.07%** |
| 70% | `E_w` | 0.3037 | **0.2575** | **15.22%** |
| 70% | `E_{x,2}^{METR}` | 2.8202 | **1.9839** | **29.65%** |

After averaging the two masks within each test graph, `E_w` and `E_{x,2}^{METR}` improve on all six test subsets at both observation levels. Their two-sided exact graph-level paired sign-flip p-value is `0.03125` at both ratios. `E_c` has weaker graph-level evidence: exact p-values are `0.0625` at 40% and `0.50` at 70%.

The six test subsets are node-disjoint but are not claimed to all be connected.

## Reproducibility

```bash
pytest -q
```

The test suite covers the directed-matrix convention, ASGC features and projection, centralized metrics, synthetic frozen-result consistency, the 380-pair candidate universe, mask nesting/partitioning, node-disjoint protocol metadata, and paper-table reconstruction.

Result-to-file provenance and SHA-256 digests are listed in [`docs/RESULT_MANIFEST.md`](docs/RESULT_MANIFEST.md). Protocol details are in [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md), and reproduction notes are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Runtime and Hardware

The implementation runs on CPU and does not require a GPU. The full node-disjoint METR-LA protocol trains graph-specific GATs for development and test scenarios and is therefore substantially heavier than the lightweight synthetic/table/figure reproduction commands. Runtime depends on hardware and the local PyTorch build.

## Citation

During review, please use:

```bibtex
@inproceedings{anonymous2027asgc,
  title     = {Aggregation-Aware Strength-Guided Calibration for Information Fusion over Incomplete Directed Graphs},
  author    = {{Anonymous authors}},
  booktitle = {Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```

## License

This repository is released under the [MIT License](LICENSE).
