#!/usr/bin/env python3
"""Generate the shared, deterministic signal batch for the feature matrix.

Every matrix cell consumes exactly the same raw float64 complex-pair files,
so cross-config comparisons are apples-to-apples. All signals stay within the
unit disk so the fixed-point (Q15/Q31) builds cannot saturate on input.

Layout:  signals/<nfft>/sig<k>.bin  -- 2*nfft native-endian float64 values
"""

import math
import os
import sys

import numpy as np

# Cover radix-2-only, mixed radix (3,5), large power of two and a non-power-of-2
# fast size. All are even, as kiss_fftr requires.
NFFTS = [64, 240, 1024, 1200]
NSIG = 4
SEED = 20261119


def build_signals(n: int) -> list[np.ndarray]:
    idx = np.arange(n, dtype=np.float64)
    two_pi = 2.0 * math.pi

    # S0: unit impulse at bin 0
    s0 = np.zeros(n, dtype=np.complex128)
    s0[0] = 1.0 + 0.0j

    # S1: single complex exponential, on-bin
    k_tone = n // 8
    s1 = np.exp(1j * two_pi * k_tone * idx / n)

    # S2: mixed tones (real cosines + imaginary sine), peak magnitude <= 1.0
    k1, k2, k3 = n // 4, 7, n // 8 + 3
    s2 = (0.5 * np.cos(two_pi * k1 * idx / n)
          + 0.3j * np.sin(two_pi * k2 * idx / n)
          + 0.2 * np.cos(two_pi * k3 * idx / n))

    # S3: seeded random complex noise
    rng = np.random.default_rng(SEED)
    s3 = rng.uniform(-0.7, 0.7, n) + 1j * rng.uniform(-0.7, 0.7, n)

    sigs = [s0, s1, s2, s3]
    for s in sigs:
        peak = np.max(np.abs(s))
        if peak > 1.0 + 1e-12:
            raise SystemExit(f"signal exceeds unit disk: peak={peak}")
    return sigs


def main(out_root: str) -> None:
    for n in NFFTS:
        d = os.path.join(out_root, str(n))
        os.makedirs(d, exist_ok=True)
        for k, sig in enumerate(build_signals(n)):
            path = os.path.join(d, f"sig{k}.bin")
            sig.astype(np.complex128).tofile(path)
        print(f"nfft={n:5d}: wrote {NSIG} signals to {d}")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "signals")
    main(os.path.abspath(root))
