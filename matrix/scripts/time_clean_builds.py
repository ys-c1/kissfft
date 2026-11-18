#!/usr/bin/env python3
"""Measure true from-scratch build time for each of the 8 matrix cells.

The main matrix build reuses CMake build dirs, so its recorded times are
incremental. This script wipes each build dir, reconfigures CMake and rebuilds
the static library with a cold cache, timing the configure+compile+link phase.
Results (seconds) go to results/clean_build_times.json.
"""

import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_matrix import BUILD, CONFIGS, RAW, RESULTS, SRC, run


def main() -> None:
    timings = []
    for cid, datatype, openmp, _flags, family, parallel in CONFIGS:
        bdir = os.path.join(BUILD, cid)
        if os.path.isdir(bdir):
            shutil.rmtree(bdir)
        os.makedirs(bdir, exist_ok=True)
        log = os.path.join(RAW, f"{cid}.cleanbuild.log")

        t0 = time.perf_counter()
        run([
            "cmake", "-G", "Unix Makefiles",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DKISSFFT_STATIC=ON",
            "-DKISSFFT_TEST=OFF",
            "-DKISSFFT_TOOLS=OFF",
            "-DKISSFFT_PKGCONFIG=OFF",
            f"-DKISSFFT_DATATYPE={datatype}",
            f"-DKISSFFT_OPENMP={'ON' if openmp else 'OFF'}",
            SRC,
        ], cwd=bdir, log_path=log)
        run(["cmake", "--build", ".", "--", "-j"], cwd=bdir, log_path=log)
        elapsed = time.perf_counter() - t0
        nproc = os.cpu_count()
        print(f"{cid:16s} clean build: {elapsed:6.2f} s "
              f"(datatype={datatype}, openmp={openmp}, -j{nproc})", flush=True)
        timings.append({"id": cid, "datatype": datatype, "openmp": openmp,
                        "family": family, "parallel": parallel,
                        "clean_build_seconds": round(elapsed, 3)})

    with open(os.path.join(RESULTS, "clean_build_times.json"), "w") as f:
        json.dump(timings, f, indent=1)
    total = sum(t["clean_build_seconds"] for t in timings)
    print(f"\ntotal clean-build time for 8 configs: {total:.2f} s")


if __name__ == "__main__":
    main()
