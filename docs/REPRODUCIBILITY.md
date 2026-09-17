# Reproducibility

## Automated tests

```bash
pytest -q
```

The tests cover the directed matrix convention, in-strength calculation, ASGC features and projection, metrics, synthetic frozen-result reconstruction, node-disjoint group roles, the 380-pair candidate universe, mask partitioning/nesting, protocol metadata, and paper-table values.

## Synthetic results

```bash
python scripts/run_synthetic.py --config configs/synthetic.yaml
python scripts/reproduce_figures.py --config configs/synthetic.yaml
```

The synthetic script recomputes Table 1 summary values from released scenario-level records and verifies per-scenario Raw-GAT/ASGC missing-edge-MAE equality. The figure script regenerates Fig. 3–5 from the released synthetic source CSVs.

## METR-LA node-disjoint experiment

A full rerun requires the prepared METR-LA files described in `docs/DATA.md`:

```bash
python scripts/run_metr_la_topology_disjoint.py
```

The script rebuilds the location-only node split, trains graph-specific GATs, performs leave-one-development-graph-out Ridge regularization selection, fits the development-only calibrator, and evaluates the six node-disjoint test subsets.

Frozen paper-facing reference outputs are under `results/frozen/metr_la_topology_disjoint/`.

## Paper tables

```bash
python scripts/reproduce_tables.py \
    --config configs/synthetic.yaml \
    --metr-config configs/metr_la.yaml
```

The table script reads only the released result artifacts and writes the paper-formatted Table 1 and Table 2 files under `results/reproduced/tables/`.

## Provenance

`docs/RESULT_MANIFEST.md` maps paper-facing tables/figures to source artifacts and SHA-256 digests. The topology-disjoint result directory also contains its own `SHA256SUMS.txt`, `protocol_manifest.json`, and `validation_audit.json`.
