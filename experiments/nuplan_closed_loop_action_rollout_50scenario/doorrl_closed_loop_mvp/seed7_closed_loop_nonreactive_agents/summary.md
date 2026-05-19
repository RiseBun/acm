# nuPlan Closed-Loop MVP Summary

nuPlan official closed-loop nonreactive simulation MVP; small scenario counts are smoke tests, not final claims.

| planner | score | no at-fault collision | drivable | making progress | progress ratio | comfort | TTC | speed limit | direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| doorrl_wm_decoupled_no_vis | 0.000 | 0.021 | 0.000 | 0.894 | 0.387 | 0.915 | 0.000 | 0.905 | 0.947 |
| doorrl_wm_object | 0.000 | 0.234 | 0.000 | 0.000 | 0.039 | 1.000 | 0.000 | 0.904 | 0.298 |

## Runner Report

| planner | succeeded | scenario | runtime mean | duration |
|---|---:|---|---:|---:|
| doorrl_wm_object | True | 00852c6784155139 | nan | 4632.0 |
| doorrl_wm_decoupled_no_vis | True | 00852c6784155139 | nan | 4758.7 |
| doorrl_wm_object | True | 06abb88ce9145a74 | nan | 4462.8 |
| doorrl_wm_decoupled_no_vis | True | 06abb88ce9145a74 | nan | 4604.3 |
| doorrl_wm_object | False | 084db2c52bb759e7 | nan | 4987.2 |
| doorrl_wm_decoupled_no_vis | False | 084db2c52bb759e7 | nan | 5178.5 |
| doorrl_wm_object | True | 0c574b7321d453a8 | nan | 4599.6 |
| doorrl_wm_decoupled_no_vis | True | 0c574b7321d453a8 | nan | 4496.4 |
| doorrl_wm_object | True | 16ebb8e8634b53c2 | nan | 4788.5 |
| doorrl_wm_decoupled_no_vis | True | 16ebb8e8634b53c2 | nan | 4484.2 |
| doorrl_wm_object | True | 1b4c481223dd52a9 | nan | 4324.2 |
| doorrl_wm_decoupled_no_vis | True | 1b4c481223dd52a9 | nan | 4397.2 |
| doorrl_wm_object | True | 1f190de308825405 | nan | 4328.4 |
| doorrl_wm_decoupled_no_vis | True | 1f190de308825405 | 0.0113 | 4181.4 |
| doorrl_wm_object | False | 23a1b30d27a05dd5 | nan | 4954.0 |
| doorrl_wm_decoupled_no_vis | False | 23a1b30d27a05dd5 | nan | 5136.1 |
| doorrl_wm_object | True | 278c7651559d593d | nan | 4801.1 |
| doorrl_wm_decoupled_no_vis | True | 278c7651559d593d | nan | 4740.9 |
| doorrl_wm_object | True | 2b73d74e9d315916 | nan | 4553.4 |
| doorrl_wm_decoupled_no_vis | True | 2b73d74e9d315916 | nan | 4605.3 |
| doorrl_wm_object | True | 31f8dbabdadc574f | nan | 4697.6 |
| doorrl_wm_decoupled_no_vis | True | 31f8dbabdadc574f | nan | 4708.3 |
| doorrl_wm_object | True | 33ee549679c55e22 | nan | 4612.3 |
| doorrl_wm_decoupled_no_vis | True | 33ee549679c55e22 | 0.0129 | 4183.3 |
| doorrl_wm_object | True | 3aeb1434949f55c0 | nan | 4515.5 |
| doorrl_wm_decoupled_no_vis | True | 3aeb1434949f55c0 | nan | 4415.9 |
| doorrl_wm_object | True | 418858b884ce5906 | nan | 4332.7 |
| doorrl_wm_decoupled_no_vis | True | 418858b884ce5906 | nan | 4588.8 |
| doorrl_wm_object | True | 4590a655a8d153ec | nan | 4733.2 |
| doorrl_wm_decoupled_no_vis | True | 4590a655a8d153ec | 0.0122 | 4409.8 |
| doorrl_wm_object | True | 4b3fe5c457f95dbc | nan | 4425.7 |
| doorrl_wm_decoupled_no_vis | True | 4b3fe5c457f95dbc | nan | 4607.6 |
| doorrl_wm_object | True | 4f6ea5cc72455dbc | nan | 4681.9 |
| doorrl_wm_decoupled_no_vis | True | 4f6ea5cc72455dbc | 0.0345 | 4104.5 |
| doorrl_wm_object | True | 50ba2ea7d7a75364 | nan | 4806.6 |
| doorrl_wm_decoupled_no_vis | True | 50ba2ea7d7a75364 | nan | 4255.2 |
| doorrl_wm_object | True | 55c0c0190c3c5386 | 0.0150 | 4204.8 |
| doorrl_wm_decoupled_no_vis | True | 55c0c0190c3c5386 | nan | 4779.0 |
| doorrl_wm_object | True | 5e86be6451225f14 | nan | 4659.7 |
| doorrl_wm_decoupled_no_vis | True | 5e86be6451225f14 | nan | 4643.0 |
| doorrl_wm_object | True | 6219380dcc3352ae | nan | 4361.9 |
| doorrl_wm_decoupled_no_vis | True | 6219380dcc3352ae | nan | 4459.9 |
| doorrl_wm_object | True | 66ea6ce04f2e573a | nan | 4582.2 |
| doorrl_wm_decoupled_no_vis | True | 66ea6ce04f2e573a | nan | 4611.3 |
| doorrl_wm_object | True | 6fe718c49b2a5df7 | nan | 4741.5 |
| doorrl_wm_decoupled_no_vis | True | 6fe718c49b2a5df7 | nan | 4453.2 |
| doorrl_wm_object | True | 75e5484fde83585b | nan | 4558.0 |
| doorrl_wm_decoupled_no_vis | True | 75e5484fde83585b | 0.0108 | 4131.7 |
| doorrl_wm_object | True | 77f3967f668359c3 | nan | 4416.9 |
| doorrl_wm_decoupled_no_vis | True | 77f3967f668359c3 | 0.0269 | 4067.3 |
| doorrl_wm_object | True | 812487a56e4c51b7 | nan | 4654.8 |
| doorrl_wm_decoupled_no_vis | True | 812487a56e4c51b7 | nan | 4781.4 |
| doorrl_wm_object | True | 830c201183415ba3 | nan | 4562.6 |
| doorrl_wm_decoupled_no_vis | True | 830c201183415ba3 | nan | 4205.7 |
| doorrl_wm_object | True | 881433d3d2f95a31 | 0.0184 | 4296.9 |
| doorrl_wm_decoupled_no_vis | True | 881433d3d2f95a31 | nan | 4468.7 |
| doorrl_wm_object | True | 8a9e60bbd8325dac | nan | 4251.8 |
| doorrl_wm_decoupled_no_vis | True | 8a9e60bbd8325dac | nan | 4687.1 |
| doorrl_wm_object | True | 8e2999f4ecb65b1b | nan | 4640.1 |
| doorrl_wm_decoupled_no_vis | True | 8e2999f4ecb65b1b | nan | 4535.5 |
| doorrl_wm_object | True | 954fc8ec046c5db4 | nan | 4904.0 |
| doorrl_wm_decoupled_no_vis | True | 954fc8ec046c5db4 | nan | 4542.0 |
| doorrl_wm_object | True | 96e3fb9e89d458fc | nan | 5015.0 |
| doorrl_wm_decoupled_no_vis | True | 96e3fb9e89d458fc | nan | 4629.5 |
| doorrl_wm_object | True | 9e37acb7e6225ade | nan | 4399.1 |
| doorrl_wm_decoupled_no_vis | True | 9e37acb7e6225ade | nan | 4923.9 |
| doorrl_wm_object | True | a335f4e5c99d5b1e | nan | 4902.4 |
| doorrl_wm_decoupled_no_vis | True | a335f4e5c99d5b1e | nan | 4424.4 |
| doorrl_wm_object | True | a6db382a1e425dc1 | nan | 4789.2 |
| doorrl_wm_decoupled_no_vis | True | a6db382a1e425dc1 | nan | 4590.8 |
| doorrl_wm_object | True | ae448a72bb2d52d1 | nan | 4353.7 |
| doorrl_wm_decoupled_no_vis | True | ae448a72bb2d52d1 | 0.0133 | 4187.9 |
| doorrl_wm_object | True | b10efeccc2905c58 | nan | 4523.0 |
| doorrl_wm_decoupled_no_vis | True | b10efeccc2905c58 | nan | 4511.9 |
| doorrl_wm_object | True | b6141e208b285df6 | nan | 4639.9 |
| doorrl_wm_decoupled_no_vis | True | b6141e208b285df6 | nan | 4804.9 |
| doorrl_wm_object | True | bb672f5e5add5d46 | nan | 4433.5 |
| doorrl_wm_decoupled_no_vis | True | bb672f5e5add5d46 | 0.0239 | 4110.8 |
| doorrl_wm_object | True | bc6d75219dea5a71 | nan | 4827.1 |
| doorrl_wm_decoupled_no_vis | True | bc6d75219dea5a71 | nan | 4788.5 |
| doorrl_wm_object | True | c07d785607005ffa | nan | 4457.4 |
| doorrl_wm_decoupled_no_vis | True | c07d785607005ffa | nan | 4594.1 |
| doorrl_wm_object | False | c3b9acf62496516f | nan | 5192.9 |
| doorrl_wm_decoupled_no_vis | False | c3b9acf62496516f | nan | 5137.1 |
| doorrl_wm_object | True | ca261de4437753ca | nan | 4615.8 |
| doorrl_wm_decoupled_no_vis | True | ca261de4437753ca | nan | 4825.1 |
| doorrl_wm_object | True | ccf23a9288075b08 | nan | 4670.3 |
| doorrl_wm_decoupled_no_vis | True | ccf23a9288075b08 | nan | 4661.3 |
| doorrl_wm_object | True | d33b4bfebc7c5870 | 0.0250 | 4373.1 |
| doorrl_wm_decoupled_no_vis | True | d33b4bfebc7c5870 | nan | 4785.3 |
| doorrl_wm_object | True | d5b11f1cce815711 | nan | 4513.4 |
| doorrl_wm_decoupled_no_vis | True | d5b11f1cce815711 | nan | 4782.3 |
| doorrl_wm_object | True | d9e8cfd282a75aff | nan | 4496.7 |
| doorrl_wm_decoupled_no_vis | True | d9e8cfd282a75aff | nan | 4715.9 |
| doorrl_wm_object | True | ddbe9d98ad3b576f | 0.0163 | 4224.8 |
| doorrl_wm_decoupled_no_vis | True | ddbe9d98ad3b576f | nan | 4614.9 |
| doorrl_wm_object | True | df7b7600a33e544d | nan | 4375.1 |
| doorrl_wm_decoupled_no_vis | True | df7b7600a33e544d | nan | 4600.1 |
| doorrl_wm_object | True | e28e969a79c6552e | nan | 4539.2 |
| doorrl_wm_decoupled_no_vis | True | e28e969a79c6552e | nan | 4897.1 |
