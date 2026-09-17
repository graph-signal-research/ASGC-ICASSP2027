# Experiment protocol

ASGC follows

`partial directed graph -> GAT preliminary edge completion -> target-specific in-strength calibration -> normalized aggregation weights -> aggregated signal`.

The eight calibration features are

`[tilde_c_i, o_i, p_i, n_i^o, n_i^m, mu_i^o, mu_i^m, sigma_i^m]`.

ASGC leaves preliminary GAT edge estimates unchanged. The calibrated in-strength is projected to `[o_i, o_i+n_i^m]`, which is valid because each candidate relation weight lies in `[0,1]`.

## Synthetic experiments

Synthetic graphs contain `N=10` nodes and `6 x 4` signals. The released artifacts cover the main comparison, theoretical-bound verification, observation-ratio analysis, missingness analysis, and the development-set ablation. Synthetic `E_x` is the paper Eq. (3) channel-weighted L1 fused-signal error.

## METR-LA node-disjoint evaluation

### Node split

Ten 20-sensor subsets are constructed from `graph_sensor_locations.csv` using deterministic Morton/Z-order grouping before adjacency loading. Four subsets are development graphs and six are test graphs. The ten subsets are mutually node-disjoint and are not required to all be connected.

### Candidate relations and masks

Every subset uses all `20*19 = 380` directed non-self pairs as candidates, including zero-weight pairs. Each graph uses two deterministic mask repetitions at 40% and 70% observation. For a given repetition the 40% observed set is nested within the 70% observed set.

Observed candidates are partitioned into fit and validation sets for graph-specific GAT training. All fit-observed candidate labels, including zeros, enter the supervised loss. Only fit-observed positive relations enter message passing. Missing candidate labels do not enter GAT fitting, validation, early stopping, checkpoint selection, or ASGC feature construction.

### Calibration training

The StandardScaler and Ridge calibrator are learned only from the four development graphs. Ridge regularization is selected by leave-one-development-graph-out cross-validation over

`[0.01, 0.1, 1, 10, 100, 1000, 10000]`.

The frozen paper run selects `lambda = 100`. After selection, the scaler and Ridge model are refit on all development graphs and frozen before evaluation on the six test graphs.

Raw-GAT and ASGC share the same preliminary completed adjacency for every test scenario. ASGC changes only the in-strength used for normalized fusion.

### Signals

The first 70% of the METR-LA speed series supplies imputation statistics and standardized 12-dimensional node descriptors based on twelve two-hour bins. The following 10% supplies 12-step speed windows ending at each evaluation timestamp and is used only for the temporal fused-signal metric.

### Statistics

For each observation ratio, the two masks are first averaged within each test graph. The six resulting graph-level paired differences are the statistical units. The release includes a 5000-resample graph-cluster bootstrap and an exact two-sided paired sign-flip test enumerating all `2^6=64` sign patterns.

## METR-LA metrics

- `E_a_all`: MAE over all hidden candidate pairs.
- `E_a_positive`: MAE over hidden candidate pairs with positive reference weight; positivity is used only during evaluation.
- `E_c`: L1 in-strength error.
- `E_w`: L1 normalized-weight error.
- `E_{x,2}^{METR}`: mean temporal L2 fused-signal deviation.
