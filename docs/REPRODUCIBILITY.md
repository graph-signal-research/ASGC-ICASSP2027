# Reproducibility

The repository includes automated consistency tests and executable reproduction paths for the reported result artifacts.

## Automated tests

```bash
pytest -q
```

The tests cover:

- directed-matrix convention `A[j,i] = j -> i`;
- in-strength calculation `c = A.T @ 1`;
- fixed METR-LA candidate-relation support of 147 nonzero directed edges;
- the eight ASGC calibration features;
- Ridge calibration and feasible projection;
- centralized edge, in-strength, weight, and signal metrics;
- development/held-out seed separation;
- GAT fit/validation/missing-edge partition isolation;
- Raw-GAT/ASGC shared edge-completion evaluation;
- synthetic scenario-level summary reconstruction;
- frozen paper-result consistency.

## Synthetic reproduction

```bash
python scripts/run_synthetic.py --config configs/synthetic.yaml
```

The script recomputes Table 1 means and sample standard deviations from `table1_per_scenario.csv`, compares them with the released summary at strict precision, and checks per-scenario Raw-GAT/ASGC missing-edge MAE equality.

## METR-LA development training

```bash
python scripts/run_metr_la_development.py --config configs/metr_la.yaml
```

With the listed environment and data, the development run refits the final ASGC StandardScaler/Ridge model from seeds `2027101–2027105` and checks its parameters against the released model using `rtol=5e-7` and `atol=5e-8`.

## METR-LA held-out replay

```bash
python scripts/evaluate_heldout_masks.py --config configs/metr_la.yaml
```

The script replays all ten held-out masks at 40% and 70% observation and compares `E_c`, `E_w`, the METR-LA temporal signal metric, and missing-edge MAE with the released per-mask CSV using `rtol=5e-7` and `atol=1e-8`; this accommodates sub-micro floating-point variation while preserving all paper-level values and significance decisions.

## Paper artifacts

```bash
python scripts/reproduce_tables.py \
    --config configs/synthetic.yaml \
    --metr-config configs/metr_la.yaml
python scripts/reproduce_figures.py --config configs/synthetic.yaml
```

The source-to-output mapping and hashes are recorded in `docs/RESULT_MANIFEST.md`.
