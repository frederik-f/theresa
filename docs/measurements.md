# Validation and performance

Measured on a 4-core machine with 11 GB RAM. Software: OGS 6.5.8–6.5.9,
ogstools 0.8.1–0.8.2, Gmsh 4.15.2, Python 3.12–3.13. Each section names the
script or command that reproduces it.

## Summary

| quantity | value |
| --- | --- |
| Duhamel construction vs. direct simulation, peak | +0.07 K |
| Surrogate vs. hold-out simulations, worst / mean absolute | 0.130 K / 0.038 K |
| Spatial discretisation error of the peak (Richardson) | −0.28 K |
| Temporal discretisation error of the peak | < 0.005 K |
| Geometric effect of depth, 300–1000 m | ≤ 0.062 K |
| Direct OGS run, reference design | 131 s |
| Surrogate prediction incl. uncertainty | ~60 ms |

## 1. Duhamel construction

`scripts/run_duhamel_check.py`. One design, assembled from a steady `ψ` solve,
a unit step-response run and the closed-form convolution with a six-term decay
curve, against a direct transient simulation with the same decay curve,
boundary temperatures and undisturbed initial field.

| | peak | at 1000 a |
| --- | --- | --- |
| direct simulation | 283.709 °C | 134.329 °C |
| Duhamel construction | 283.779 °C | 134.621 °C |
| **difference** | **+0.0705 K** | +0.292 K |

The largest difference over the record is 0.498 K at about 100 a, during the
steepest decline.

## 2. Surrogate accuracy

`theresa validate --points 20`. Hold-out designs from a Latin hypercube with a
different seed and an inner margin of 5 %.

Training set: 400 Latin-hypercube designs over the ten interpolated variables
(two spacings; conductivity and volumetric heat capacity of four materials),
839 min, reduced to 8 kernel modes.

| | value |
| --- | --- |
| worst error | **0.130 K** |
| mean absolute error | 0.038 K |
| bias | −0.004 K |
| range of simulated peaks | 71–140 °C |
| largest probe-grid deficit | 0.299 % |

The hold-out comparison uses the same discretisation on both sides and
measures the interpolation error only.

## 3. Depth

`scripts/run_depth_study.py`. Peak temperature rise at fixed spacings after
removing the analytic depth term `g·ψ`.

| pitch | drift spacing | 300 m | 500 m | 700 m | 1000 m | spread |
| --- | --- | --- | --- | --- | --- | --- |
| 10 m | 30 m | 279.94 K | 279.90 K | 279.88 K | 279.90 K | 0.062 K |
| 15 m | 60 m | 237.48 K | 237.49 K | 237.47 K | 237.47 K | 0.019 K |
| 25 m | 100 m | 227.72 K | 227.72 K | 227.76 K | 227.75 K | 0.044 K |

Depth therefore enters analytically. The domain keeps a constant clearance of
700 m below the drift. Depths below 300 m are flagged by `predict`.

## 4. Discretisation

Refinement sequences at the reference design with ratio √2; element sizes
scaled uniformly, time schedule scaled self-similarly. Quantity: peak
temperature at the reference loading.

**Spatial**

| scale | fine size | nodes | peak | change |
| --- | --- | --- | --- | --- |
| 2.000 | 0.300 m | | 72.1558 °C | |
| 1.414 | 0.212 m | | 72.5472 °C | +0.3914 K |
| **1.000** | **0.150 m** | ~15 k | **72.8771 °C** | +0.3298 K |
| 0.707 | 0.106 m | 43 067 | 73.0297 °C | +0.1527 K |

Observed order p = 2.22, Richardson estimate 73.1612 °C. The shipped mesh is
0.284 K below the extrapolated value; convergence is monotone from below.

**Temporal**

| scale | steps | peak | change |
| --- | --- | --- | --- |
| 2.000 | 197 | 72.8749 °C | |
| 1.414 | 237 | 72.8731 °C | −0.0018 K |
| **1.000** | **296** | **72.8771 °C** | +0.0039 K |
| 0.707 | 379 | 72.8748 °C | −0.0023 K |

Changes alternate in sign within 0.004 K (solver noise level).

The spatial error is systematic and common to all training runs. It largely
cancels in comparisons between designs; for an absolute value close to a
temperature limit, a direct run on a refined mesh is recommended.

**Convolution on the time schedule** (`tests/test_duhamel.py`, analytic
exponential-sum kernel):

| schedule | steps | distinct Δt | worst error | error at peak |
| --- | --- | --- | --- | --- |
| Δt₀ = 5e3 s, ×1.4, max 2 a | 603 | 31 | 0.045 K | 0.011 K |
| **Δt₀ = 5e3 s, ×1.5, max 5 a** | **295** | **28** | **0.099 K** | **0.017 K** |
| Δt₀ = 5e3 s, ×1.6, max 5 a | 282 | 27 | 0.135 K | 0.020 K |

## 5. Source term and probe grid

The source density refers to the meshed canister volume, so the injected power
equals the specified power (production mesh, pitch 8 m, drift spacing 25 m):

| reference volume | volume | injected per watt canister power |
| --- | --- | --- |
| exact cylinder | 2.637656 m³ | 0.24887292 W |
| **meshed quarter** | **2.625765 m³** | **0.25000000 W** |

Meshed/exact volume ratio 0.99549; it converges to 1 under refinement
(`tests/test_geometry.py`).

The probe grid (156 points, 2 cm inside the surface) differs from the exact
nodal wall maximum by −0.013 % at the reference design. The deficit is recorded
for every training run (`TrainingSet.probe_deficits`).

## 6. Sensitivity

`theresa sobol`, N = 4096, scrambled Sobol' sequence, Saltelli/Jansen
estimators, bootstrap standard errors. Inputs uniform over the training
ranges; geothermal gradient 0.02–0.07 K/m, surface temperature 6–14 °C.

Layout fixed (mean 80.2 °C, standard deviation 16.9 K):

| input | S_i | S_Ti |
| --- | --- | --- |
| **backfill λ** | **0.715 ± 0.098** | 0.715 ± 0.014 |
| geothermal gradient | 0.182 ± 0.045 | 0.182 ± 0.004 |
| host rock λ | 0.077 ± 0.028 | 0.077 ± 0.002 |
| surface temperature | 0.019 ± 0.015 | 0.019 ± 0.000 |
| other six | ≤ 0.003 | ≤ 0.003 |

Spacings varied as well (mean 84.2 °C, standard deviation 18.0 K): backfill λ
0.564 ± 0.091, gradient 0.161, host rock λ 0.094, clear canister spacing
0.066, drift spacing 0.058.

- The backfill conductivity is the dominant input: the full canister power
  crosses the low-conductivity bentonite shell.
- Volumetric heat capacities are negligible (S_i ≤ 0.005): at the time of the
  peak (about 5 a) the near field is quasi-steady.
- Interactions are small: S_Ti − S_i ≤ 0.023; interaction share 0.000 (layout
  fixed) and 0.030 (spacings varied).

The indices refer to the stated input distributions.

## 7. Loading and interim storage

Fully loaded canister (ten PWR fuel elements), drift 3.5 m, bentonite
λ = 0.8 W/(m·K); one step-response run, convolved for several interim storage
times:

| interim storage | canister power | peak | at | at 1000 a |
| --- | --- | --- | --- | --- |
| 30 a | 8347 W | 343.5 °C | 4.7 a | 107.5 °C |
| 50 a | 6175 W | 262.6 °C | 5.1 a | 102.9 °C |
| 100 a | 3429 W | 160.8 °C | 8.6 a | 95.1 °C |
| 200 a | 1945 W | 112.1 °C | 113.4 a | 85.6 °C |
| 300 a | 1554 W | 99.3 °C | 145.4 a | 78.7 °C |
| 500 a | 1146 W | 81.0 °C | 167.4 a | 68.4 °C |

For short storage the peak occurs after about 5 a and is governed by the near
field; for long storage after about 150 a, governed by the array interaction.
The reference configuration (two fuel elements, 1235 W at 50 a) peaks at
72.7 °C after 5.3 a; admissible loading for 100 °C: 1942 W.

## 8. Performance

**Direct run** (`pipeline.run_case`, reference design, idle machine):

| stage | time |
| --- | --- |
| meshing: geometry, curves, surfaces, volume | 16 s |
| meshing: conversion to OGS VTU | 5 s |
| steady `ψ` solve | 4 s |
| transient solve: linear solver | 73 s |
| transient solve: output | 7 s |
| transient solve: assembly, reading, setup | 21 s |
| step-response extraction | 4 s |
| **total** | **131 s** |

**Linear solver** (17 k nodes, per time step): CG with diagonal
preconditioner 0.86 s, Eigen SparseLU 9.0 s.

**CG tolerance** (full 1000 a record):

| tolerance | solver time | largest change of the step response | at 1235 W |
| --- | --- | --- | --- |
| 1e-14 | 90 s | — | — |
| **1e-10** | 73 s | 1.0e-11 K/W | 1.3e-8 K |
| 1e-8 | 67 s | 9.5e-10 K/W | 1.2e-6 K |

Transient solves use 1e-10, the steady solve 1e-14.

**Parallel runs.** The solver is memory-bandwidth bound: three concurrent runs
on four cores each take 3.0× as long per step as a single run. OGS in the
`ogs` wheel runs single-threaded; `OMP_NUM_THREADS` has no effect on the solve
time.

**Mesh conversion.** `mesh_conversion.convert_mesh` derives bulk node and
element ids from the extraction indices and a sorted tetrahedron face table.
Output files are identical to ogstools array by array
(`tests/test_mesh_conversion.py`).
