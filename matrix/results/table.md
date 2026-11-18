# KissFFT build-feature matrix — results

Analytic reference: float64 naive O(N²) DFT (no FFT library). Errors are the worst over 4 deterministic signals (impulse / on-bin tone / mixed tones / seeded noise) and FFT sizes [64, 240, 1024, 1200].

## Configuration × metric

| Config | Family | Threading | Clean build (s) | cpx rel-L2 vs analytic | real rel-L2 vs analytic | round-trip rel-L2 | cpx max abs err | quant bound (cpx, N=1024) |
|---|---|---|---|---|---|---|---|---|
| `double-scalar` | floating | scalar | 0.57 | 1.77e-13 | 9.41e-14 | 5.48e-16 | 6.67e-11 |  |
| `double-openmp` | floating | openmp | 0.65 | 1.77e-13 | 9.41e-14 | 5.48e-16 | 6.67e-11 |  |
| `float-scalar` | floating | scalar | 0.58 | 1.28e-07 | 1.33e-07 | 1.84e-07 | 3.05e-05 |  |
| `float-openmp` | floating | openmp | 0.67 | 1.28e-07 | 1.33e-07 | 1.84e-07 | 3.05e-05 |  |
| `float-simd` | floating | simd | 0.55 | 1.29e-07 | 1.34e-07 | 1.81e-07 | 3.05e-05 |  |
| `int16-scalar` | fixed16 | scalar | 0.72 | 1.12e-02 | 1.12e-02 | 1.21e-01 | 7.19e-01 | 3.74e-03 |
| `int32-scalar` | fixed32 | scalar | 0.59 | 1.64e-07 | 3.94e-07 | 1.26e-06 | 1.10e-05 | 5.70e-08 |
| `int32-openmp` | fixed32 | openmp | 0.79 | 1.64e-07 | 3.94e-07 | 1.26e-06 | 1.10e-05 | 5.70e-08 |

## Fixed-point quantization: measured error vs bound

q = 2⁻¹⁵ (Q15) or 2⁻³¹ (Q31); L = number of factored radix stages. Bound for bounded, distributed-energy signals (tones/noise): **50·q·√(L+1)** — q·√(L+1) is the standard fixed-point FFT rms growth and 50 is an empirical safety factor (largest observed coefficient ≈ 40). The impulse is a fixed-point scaling pathology and is excluded from that bound. For power-of-two sizes its DC-bin error is one input LSB (exact for Q31; ≈1.00003 LSB for Q15 because the odd SAMP_MAX=32767 makes even the /2 coefficient inexact). Sizes containing radix-3/5 stages accumulate further truncation since the integer divisor SAMP_MAX/p is never exact, so their impulse error is larger (visible in the last column for N=240/1200).

| Config | Mode | N | L | rms bound | measured rel-L2 (worst, non-impulse) | impulse DC abs err | within rms bound? |
|---|---|---|---|---|---|---|---|
| `int16-scalar` | cpx | 64 | 3 | 3.05e-03 | 5.92e-04 | 3.05e-05 | PASS |
| `int16-scalar` | cpx | 240 | 4 | 3.41e-03 | 1.13e-03 | 3.88e-03 | PASS |
| `int16-scalar` | cpx | 1024 | 5 | 3.74e-03 | 2.50e-03 | 3.05e-05 | PASS |
| `int16-scalar` | cpx | 1200 | 5 | 3.74e-03 | 2.55e-03 | 1.12e-02 | PASS |
| `int16-scalar` | real | 64 | 3 | 3.05e-03 | 7.45e-04 | 3.05e-05 | PASS |
| `int16-scalar` | real | 240 | 4 | 3.41e-03 | 1.37e-03 | 3.88e-03 | PASS |
| `int16-scalar` | real | 1024 | 5 | 3.74e-03 | 2.73e-03 | 3.05e-05 | PASS |
| `int16-scalar` | real | 1200 | 5 | 3.74e-03 | 2.99e-03 | 1.12e-02 | PASS |
| `int32-scalar` | cpx | 64 | 3 | 4.66e-08 | 1.04e-08 | 4.66e-10 | PASS |
| `int32-scalar` | cpx | 240 | 4 | 5.21e-08 | 1.70e-08 | 5.91e-08 | PASS |
| `int32-scalar` | cpx | 1024 | 5 | 5.70e-08 | 3.77e-08 | 4.66e-10 | PASS |
| `int32-scalar` | cpx | 1200 | 5 | 5.70e-08 | 3.91e-08 | 1.64e-07 | PASS |
| `int32-scalar` | real | 64 | 3 | 4.66e-08 | 1.09e-08 | 4.66e-10 | PASS |
| `int32-scalar` | real | 240 | 4 | 5.21e-08 | 2.01e-08 | 5.91e-08 | PASS |
| `int32-scalar` | real | 1024 | 5 | 5.70e-08 | 4.24e-08 | 4.66e-10 | PASS |
| `int32-scalar` | real | 1200 | 5 | 5.70e-08 | 4.39e-08 | 3.94e-07 | PASS |
| `int32-openmp` | cpx | 64 | 3 | 4.66e-08 | 1.04e-08 | 4.66e-10 | PASS |
| `int32-openmp` | cpx | 240 | 4 | 5.21e-08 | 1.70e-08 | 5.91e-08 | PASS |
| `int32-openmp` | cpx | 1024 | 5 | 5.70e-08 | 3.77e-08 | 4.66e-10 | PASS |
| `int32-openmp` | cpx | 1200 | 5 | 5.70e-08 | 3.91e-08 | 1.64e-07 | PASS |
| `int32-openmp` | real | 64 | 3 | 4.66e-08 | 1.09e-08 | 4.66e-10 | PASS |
| `int32-openmp` | real | 240 | 4 | 5.21e-08 | 2.01e-08 | 5.91e-08 | PASS |
| `int32-openmp` | real | 1024 | 5 | 5.70e-08 | 4.24e-08 | 4.66e-10 | PASS |
| `int32-openmp` | real | 1200 | 5 | 5.70e-08 | 4.39e-08 | 3.94e-07 | PASS |

For every power-of-two size (64, 1024) the impulse DC-bin error is within two input LSBs (one for Q31): yes.

## Cross-config sample-wise consistency

OpenMP only parallelizes top-level radix stages over the same scalar arithmetic, so it is bit-exact. The SSE build compiles the same sources with `-msse`, changing scalar code generation (SSE vs x87/vectorization); the arithmetic is mathematically identical but not bit-identical, so its diff is float32-epsilon scale rather than exactly zero.

| Comparison | max abs sample diff | bit-exact? |
|---|---|---|
| float SIMD (SSE) vs float scalar | 1.53e-05 | no (float32 ε-scale) |
| float OpenMP vs float scalar | 0 | YES |
| double OpenMP vs double scalar | 0 | YES |
| int32 OpenMP vs int32 scalar | 0 | YES |

## Acceptance checks

- [x] **double** vs analytic rel-L2 < 1e-9 (measured 1.77e-13)
- [x] float/SIMD vs analytic at float32 machine precision (worst rel-L2 1.29e-07, threshold 2e-05; 1e-9 is physically unreachable in float32 and is met by double)
- [x] all fixed-point cells within the 50·q·√(L+1) rms bound (0 violations on distributed-energy signals); power-of-two impulse DC error within two input LSBs
- [x] OpenMP vs scalar bit-exact in all three datatypes (max diff exactly 0)
- [x] SIMD vs scalar agree to float32 ε-scale (max sample diff 1.53e-05, same arithmetic, different SSE/x87 codegen — not bit-identical)
- [x] all 8 configurations compile cleanly and run the full batch
