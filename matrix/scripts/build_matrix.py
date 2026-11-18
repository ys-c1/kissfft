#!/usr/bin/env python3
"""Build the 8-cell feature matrix and run the shared signal batch on each.

Configurations (datatype x threading):
  double   scalar / openmp
  float    scalar / openmp / simd(SSE)
  int16_t  scalar
  int32_t  scalar / openmp

Each cell gets its own CMake static-library build. The harness is compiled
against that library with matching preprocessor definitions, then all
(mode, nfft, signal) combinations are executed. Raw metrics land in
results/raw/<cell>.json; analyze.py turns them into the final table.
"""

import json
import os
import subprocess
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.abspath(os.path.join(ROOT, ".."))          # kissfft source tree
BUILD = os.path.join(ROOT, "build")
RESULTS = os.path.join(ROOT, "results")
RAW = os.path.join(RESULTS, "raw")
SIGNALS = os.path.join(ROOT, "signals")
HARNESS_SRC = os.path.join(ROOT, "harness", "matrix_harness.c")

NFFTS = [64, 240, 1024, 1200]
MODES = ["cpx", "real", "rt"]

# id, cmake datatype, openmp, extra harness compile flags, feature axis labels
CONFIGS = [
    ("double-scalar",  "double",  False, [],                "floating", "scalar"),
    ("double-openmp",  "double",  True,  ["-fopenmp"],      "floating", "openmp"),
    ("float-scalar",   "float",   False, [],                "floating", "scalar"),
    ("float-openmp",   "float",   True,  ["-fopenmp"],      "floating", "openmp"),
    ("float-simd",     "simd",    False, ["-msse", "-DUSE_SIMD"],
                                                                  "floating", "simd"),
    ("int16-scalar",   "int16_t", False, [],                "fixed16",  "scalar"),
    ("int32-scalar",   "int32_t", False, [],                "fixed32",  "scalar"),
    ("int32-openmp",   "int32_t", True,  ["-fopenmp"],      "fixed32",  "openmp"),
]

DATATYPE_DEFS = {
    "double":  ["-Dkiss_fft_scalar=double"],
    "float":   ["-Dkiss_fft_scalar=float"],
    "int16_t": ["-DFIXED_POINT=16"],
    "int32_t": ["-DFIXED_POINT=32"],
    "simd":    ["-DUSE_SIMD"],
}


def run(cmd, cwd=None, log_path=None):
    if log_path:
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, cwd=cwd, stdout=logf,
                                  stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}"
                           + (f"\nsee {log_path}" if log_path else ""))


def build_cell(cid, datatype, openmp, harness_flags):
    bdir = os.path.join(BUILD, cid)
    os.makedirs(bdir, exist_ok=True)
    log = os.path.join(RAW, f"{cid}.build.log")

    t0 = time.perf_counter()
    cmake = [
        "cmake", "-G", "Unix Makefiles",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DKISSFFT_STATIC=ON",
        "-DKISSFFT_TEST=OFF",
        "-DKISSFFT_TOOLS=OFF",
        "-DKISSFFT_PKGCONFIG=OFF",
        f"-DKISSFFT_DATATYPE={datatype}",
        f"-DKISSFFT_OPENMP={'ON' if openmp else 'OFF'}",
        SRC,
    ]
    run(cmake, cwd=bdir, log_path=log)
    run(["cmake", "--build", ".", "--", "-j"], cwd=bdir, log_path=log)
    build_s = time.perf_counter() - t0

    libs = [f for f in os.listdir(bdir)
            if f.startswith("libkissfft-") and f.endswith(".a")]
    if len(libs) != 1:
        raise RuntimeError(f"expected exactly one static lib in {bdir}, got {libs}")
    lib = os.path.join(bdir, libs[0])

    # Compile the harness with exactly the datatype definition the library was
    # built with (CMake exposes them PUBLIC, but we mirror them explicitly).
    hbin = os.path.join(bdir, "matrix_harness")
    hlog = os.path.join(RAW, f"{cid}.harness.log")
    cc = ["gcc", "-O2", "-std=c11", "-I", SRC]
    if openmp:
        cc.append("-fopenmp")
    cc += [*DATATYPE_DEFS[datatype], *harness_flags,
           HARNESS_SRC, lib, "-lm", "-o", hbin]
    run(cc, log_path=hlog)
    return build_s, hbin


def run_cell(hbin):
    """Run every (mode, nfft, signal) combo; return run manifest."""
    runs = []
    for nfft in NFFTS:
        for sig in range(4):
            inp = os.path.join(SIGNALS, str(nfft), f"sig{sig}.bin")
            for mode in MODES:
                if mode in ("real", "rt") and (nfft & 1):
                    continue
                out = os.path.join(RAW, f"out_{os.path.basename(os.path.dirname(hbin))}"
                                        f"_{mode}_{nfft}_s{sig}.txt")
                t0 = time.perf_counter()
                run([hbin, mode, str(nfft), inp, out])
                wall = time.perf_counter() - t0
                runs.append({"mode": mode, "nfft": nfft, "sig": sig,
                             "out": out, "wall_s": wall})
    return runs


def main():
    os.makedirs(RAW, exist_ok=True)
    summary = []
    for cid, datatype, openmp, hflags, family, parallel in CONFIGS:
        print(f"=== {cid} ({datatype}, openmp={openmp}) ===", flush=True)
        build_s, hbin = build_cell(cid, datatype, openmp, hflags)
        print(f"    build+link: {build_s:.2f} s", flush=True)
        runs = run_cell(hbin)
        rec = {"id": cid, "datatype": datatype, "openmp": openmp,
               "family": family, "parallel": parallel,
               "build_seconds": round(build_s, 3),
               "n_runs": len(runs), "runs": runs}
        with open(os.path.join(RAW, f"{cid}.json"), "w") as f:
            json.dump(rec, f, indent=1)
        summary.append(rec)
    with open(os.path.join(RAW, "_index.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "runs"} for r in summary],
                  f, indent=1)
    print(f"\nAll {len(CONFIGS)} cells built. Raw data in {RAW}")


if __name__ == "__main__":
    main()
