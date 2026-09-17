# Node-disjoint METR-LA evaluation

This is the real-data protocol used in the paper.

## Split and candidate construction

- The 207 METR-LA sensors are ordered using only latitude/longitude from `graph_sensor_locations.csv` with deterministic 16-bit Morton/Z-order.
- Consecutive groups of 20 define ten mutually node-disjoint subsets; four are development subsets and six are test subsets. Seven sensors remain unused.
- The split is fixed before the adjacency matrix is loaded.
- Every 20-node subset uses all 380 directed non-self node pairs as candidate relations, independently of the reference adjacency.
- The six test subsets are node-disjoint but are not claimed to all be connected.

## GAT isolation

Each graph/mask/observation-ratio scenario trains a fresh DenseGAT. The supervised fit loss uses only fit-observed candidate labels, including observed zeros. Message passing uses only fit-observed positive relations. Hidden candidate labels remain unavailable to GAT fitting, validation, early stopping, and ASGC feature construction.

## Calibration isolation

The StandardScaler and Ridge calibration model are fitted exclusively from the four development graphs. Ridge regularization is chosen by leave-one-development-graph-out cross-validation. The frozen run selects `lambda = 100` from `[0.01, 0.1, 1, 10, 100, 1000, 10000]`. The fitted calibrator is frozen before test calibration.

## Temporal isolation

The first 70% of the METR-LA signal series supplies imputation statistics and 12-dimensional node descriptors. The next 10% supplies 12-step windows ending at evaluation timestamps and is used only for the temporal fusion metric. The remaining 20% is outside the evaluation reader.

## Frozen paper results

| Observation | Metric | Raw-GAT | ASGC | Relative reduction |
|---:|---|---:|---:|---:|
| 40% | `E_c` | 20.081425 | 14.864944 | 25.98% |
| 40% | `E_w` | 0.532109 | 0.382059 | 28.20% |
| 40% | `E_x2_METR` | 4.878534 | 2.905146 | 40.45% |
| 70% | `E_c` | 10.736535 | 9.870400 | 8.07% |
| 70% | `E_w` | 0.303707 | 0.257489 | 15.22% |
| 70% | `E_x2_METR` | 2.820166 | 1.983881 | 29.65% |

After averaging the two masks within each test graph, `E_w` and `E_x2_METR` improve on all six test graphs at both ratios. Their two-sided exact graph-level sign-flip p-value is `0.03125`. `E_c` is less uniform: exact p-values are `0.0625` at 40% and `0.50` at 70%.

Source artifacts and hashes are stored under `results/frozen/metr_la_topology_disjoint/`.
