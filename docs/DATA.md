# Data

## Synthetic experiments

Synthetic paper artifacts are included under `results/frozen/synthetic/` and do not require external data.

## METR-LA

Raw METR-LA measurements are not redistributed. Prepare the public DCRNN/METR-LA files under `data/METR-LA/`:

```text
adj_mx.pkl
distances_la_2012.csv
graph_sensor_ids.txt
graph_sensor_locations.csv
metr-la.h5
```

The helper script downloads and SHA-256 verifies the four sensor-graph files:

```bash
python scripts/download_metr_la.py --config configs/metr_la.yaml
python scripts/download_metr_la.py --config configs/metr_la.yaml \
    --metr-h5 /path/to/metr-la.h5
```

Reference SHA-256 values:

- `adj_mx.pkl`: `a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3`
- `distances_la_2012.csv`: `a576a2a3e28dbb959be6da22688e24dd1b246b81264595e129147c256cd53de5`
- `graph_sensor_ids.txt`: `3ba026caa2e6263ab0ea54b0fa1b125dbfa7216544cd05313b555e826292b990`
- `graph_sensor_locations.csv`: `eb8ea96e07358b45d0e4ba3b89c2673fa20c54af50150249e627389e749ade6f`

The reference `metr-la.h5` used for the reported run has SHA-256
`64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05`
and shape `34272 x 207`.

### Matrix convention

The official adjacency is used without transposition after removing self-loops:

```text
A[j,i] = j -> i
c = A.T @ ones(N)
```

### Node-disjoint split

The paper experiment fixes ten 20-sensor subsets from `graph_sensor_locations.csv` only, using deterministic 16-bit Morton/Z-order grouping. The split is fixed before `adj_mx.pkl` is loaded. Groups `0,3,6,9` are development subsets and groups `1,2,4,5,7,8` are test subsets. All ten subsets are mutually node-disjoint; seven sensors remain unused.

Each subset uses every directed non-self pair as a candidate relation, giving `20 x 19 = 380` candidates independently of the reference adjacency. Candidate support therefore does not expose which test pairs have positive reference weight.

### Temporal protocol

The experiment reads only the first 80% of the speed series:

- first 70%: training-side imputation statistics and 12-dimensional node descriptors;
- next 10%: 12-step windows ending at evaluation timestamps, used only for the temporal fusion metric;
- final 20%: outside the evaluation reader.

All imputation statistics and descriptor standardization are fit from the first 70% only.
