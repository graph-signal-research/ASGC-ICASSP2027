# Data protocol

## METR-LA

The repository does not redistribute `metr-la.h5`. Obtain METR-LA from the original DCRNN data release and place the required files under `data/METR-LA/`.

Original DCRNN source: https://github.com/liyaguang/DCRNN

Required files: `metr-la.h5`, `adj_mx.pkl`, `graph_sensor_ids.txt`, `distances_la_2012.csv`, `graph_sensor_locations.csv`.

### Direction

The official adjacency is used directly: `A_paper = adj_mx.copy()`, then self-loops are removed. No transpose is performed. In paper notation, `A[j,i] = j -> i`, and target-node in-strength is `c = A.T @ ones(N)`.

### Fixed 20 sensors

The ordered IDs are stored in `results/frozen/metr_la/development/selected_sensor_ids.txt`; SHA-256 is `692779f867902d401354b31816aaf6bbf5d42c57f6a1f056dce9da89a9332bd7`. The resulting subgraph contains 147 nonzero directed, non-self candidate relations. Observation masks are sampled only within this fixed candidate support; zero entries are not treated as missing links.

### Reference file hashes

The download helper verifies the four DCRNN sensor-graph files against the SHA-256 values used in this release:

- `adj_mx.pkl`: `a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3`
- `distances_la_2012.csv`: `a576a2a3e28dbb959be6da22688e24dd1b246b81264595e129147c256cd53de5`
- `graph_sensor_ids.txt`: `3ba026caa2e6263ab0ea54b0fa1b125dbfa7216544cd05313b555e826292b990`
- `graph_sensor_locations.csv`: `eb8ea96e07358b45d0e4ba3b89c2673fa20c54af50150249e627389e749ade6f`

The reference `metr-la.h5` used for the reported runs has shape `34272 x 207`. The loader validates the sensor ordering, dimensions, and five-minute timestamp spacing.

### Temporal protocol

The public evaluation reader loads the first 80% of the time series: the first 70% provides training-side preprocessing statistics and the following 10% provides temporal evaluation windows.
