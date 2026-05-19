# H=10 Capacity Latency Benchmark: All Compact Variants

Source: synthetic batch size 128, horizon 10.

| K | Variant | Forward ms | H=10 rollout ms | Peak fwd MB | Peak H=10 rollout MB |
|---:|---|---:|---:|---:|---:|
| 16 | holistic_query | 2.631 | 49.263 | 69.6 | 107.4 |
| 16 | object_only | 2.547 | 50.285 | 68.3 | 110.6 |
| 16 | shared_objrel | 2.361 | 46.805 | 68.3 | 110.6 |
| 16 | door | 2.937 | 53.509 | 68.5 | 110.9 |
| 32 | holistic_query | 2.692 | 48.295 | 72.9 | 164.4 |
| 32 | object_only | 2.675 | 48.843 | 68.3 | 166.6 |
| 32 | shared_objrel | 2.503 | 47.669 | 68.3 | 166.6 |
| 32 | door | 2.920 | 54.004 | 68.5 | 166.8 |
| 64 | holistic_query | 2.665 | 49.767 | 89.0 | 286.2 |
| 64 | object_only | 2.580 | 50.474 | 77.9 | 288.1 |
| 64 | shared_objrel | 2.443 | 47.649 | 77.8 | 288.1 |
| 64 | door | 2.923 | 53.385 | 83.1 | 288.3 |
