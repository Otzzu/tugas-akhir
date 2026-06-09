# CPG Diameter Analysis (megavul.hdf5)

Undirected approx diameter (2-sweep BFS, largest CC). Flat GNN reach = 4 hops → **diameter > 4 = over-squashing risk**.

### megavul / all

| Metric | Value |
| --- | --- |
| Count | 31016 |
| Mean | 5.4 |
| Median (P50) | 5 |
| P90 | 7 |
| P95 | 7 |
| P99 | 7 |
| Max | 13 |
| diameter > 4 | 30992 (99.9%) |
| diameter > 6 | 3920 (12.6%) |
| diameter > 8 | 32 (0.1%) |
| diameter > 10 | 10 (0.0%) |
| diameter > 16 | 0 (0.0%) |

| Node bucket | Count | Mean diam | % diam>4 |
| --- | --- | --- | --- |
| [0, 100) | 8172 | 5.5 | 100% |
| [100, 200) | 7534 | 5.4 | 100% |
| [200, 500) | 9793 | 5.3 | 100% |
| [500, 1000) | 3810 | 5.2 | 100% |
| [1000, inf) | 1707 | 5.2 | 100% |

### megavul / benign

| Metric | Value |
| --- | --- |
| Count | 12500 |
| Mean | 5.3 |
| Median (P50) | 5 |
| P90 | 7 |
| P95 | 7 |
| P99 | 7 |
| Max | 13 |
| diameter > 4 | 12498 (100.0%) |
| diameter > 6 | 1432 (11.5%) |
| diameter > 8 | 12 (0.1%) |
| diameter > 10 | 5 (0.0%) |
| diameter > 16 | 0 (0.0%) |

| Node bucket | Count | Mean diam | % diam>4 |
| --- | --- | --- | --- |
| [0, 100) | 5134 | 5.5 | 100% |
| [100, 200) | 3913 | 5.3 | 100% |
| [200, 500) | 3048 | 5.2 | 100% |
| [500, 1000) | 401 | 5.3 | 100% |
| [1000, inf) | 4 | 5.0 | 100% |

### megavul / vulnerable

| Metric | Value |
| --- | --- |
| Count | 18516 |
| Mean | 5.4 |
| Median (P50) | 5 |
| P90 | 7 |
| P95 | 7 |
| P99 | 7 |
| Max | 13 |
| diameter > 4 | 18494 (99.9%) |
| diameter > 6 | 2488 (13.4%) |
| diameter > 8 | 20 (0.1%) |
| diameter > 10 | 5 (0.0%) |
| diameter > 16 | 0 (0.0%) |

| Node bucket | Count | Mean diam | % diam>4 |
| --- | --- | --- | --- |
| [0, 100) | 3038 | 5.6 | 99% |
| [100, 200) | 3621 | 5.4 | 100% |
| [200, 500) | 6745 | 5.3 | 100% |
| [500, 1000) | 3409 | 5.2 | 100% |
| [1000, inf) | 1703 | 5.2 | 100% |
