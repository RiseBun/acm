# Capacity Latency Benchmark

| K | Variant | Forward ms | H=20 rollout ms | Peak mem fwd MB | Peak mem rollout MB |
|---:|---|---:|---:|---:|---:|
| 16 | holistic_query | 2.621 | 98.328 | 69.6 | 118.1 |
| 16 | object_only | 2.758 | 101.040 | 68.3 | 121.7 |
| 16 | shared_objrel | 2.600 | 104.731 | 68.3 | 121.7 |
| 16 | door | 3.316 | 109.811 | 68.5 | 121.9 |
| 32 | holistic_query | 2.668 | 98.178 | 72.9 | 185.9 |
| 32 | object_only | 3.265 | 99.673 | 68.3 | 187.9 |
| 32 | shared_objrel | 2.480 | 95.808 | 68.3 | 187.9 |
| 32 | door | 3.023 | 107.940 | 68.5 | 188.0 |
| 64 | holistic_query | 2.744 | 99.403 | 89.0 | 327.4 |
| 64 | object_only | 2.694 | 100.176 | 77.9 | 329.2 |
| 64 | shared_objrel | 2.559 | 100.591 | 77.8 | 329.2 |
| 64 | door | 3.087 | 109.195 | 83.1 | 329.5 |
| 97 | full_holistic_97 | 2.368 | 92.543 | 88.2 | 470.9 |
