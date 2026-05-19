# Stage 1 Pilot X: 3-seed verification of v3 ranking

Seeds: [7, 42, 123]  Conditions: ['wm_object', 'wm_decoupled', 'wm_decoupled_no_vis']


| condition | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---|---|---|---|---|---|
| wm_object | 31.790 ± 19.700 | 0.597 ± 0.048 | 0.610 ± 0.048 | 0.258 ± 0.058 | 0.170 ± 0.034 |
| wm_decoupled | 4.337 ± 13.661 | 0.695 ± 0.283 | 0.676 ± 0.243 | 0.636 ± 0.108 | 0.056 ± 0.001 |
| wm_decoupled_no_vis | 0.342 ± 15.969 | 0.820 ± 0.260 | 0.814 ± 0.167 | 0.223 ± 0.033 | 0.043 ± 0.015 |

## Per-seed raw values

### wm_object

| seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---|---|---|---|---|---|
| 7 | 21.788 | 0.602 | 0.626 | 0.193 | 0.135 |
| 42 | 54.484 | 0.547 | 0.556 | 0.302 | 0.203 |
| 123 | 19.097 | 0.643 | 0.649 | 0.279 | 0.171 |

### wm_decoupled

| seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---|---|---|---|---|---|
| 7 | -1.454 | 0.837 | 0.811 | 0.727 | 0.056 |
| 42 | 19.940 | 0.369 | 0.395 | 0.517 | 0.057 |
| 123 | -5.475 | 0.878 | 0.821 | 0.666 | 0.056 |

### wm_decoupled_no_vis

| seed | Return | CollRate | CollMean | Stab(ego-cos) | Stab(L2) |
|---|---|---|---|---|---|
| 7 | -9.786 | 0.970 | 0.950 | 0.187 | 0.029 |
| 42 | 18.750 | 0.970 | 0.864 | 0.253 | 0.042 |
| 123 | -7.938 | 0.520 | 0.628 | 0.228 | 0.059 |
