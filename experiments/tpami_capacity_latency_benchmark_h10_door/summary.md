# DOOR H=10 Capacity Latency Benchmark

Source: synthetic batch size 128, same DOOR splits as Stage-1 K-scaling.

| K | Split | Forward ms | H=10 rollout ms | Peak fwd MB | Peak H=10 rollout MB |
|---:|---:|---:|---:|---:|---:|
| 16 | 14/2 | 3.119 | 55.599 | 68.5 | 110.9 |
| 32 | 24/8 | 2.969 | 56.429 | 68.5 | 166.8 |
| 64 | 52/12 | 3.159 | 55.416 | 83.1 | 288.3 |
