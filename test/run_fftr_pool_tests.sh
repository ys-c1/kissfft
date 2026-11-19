#!/usr/bin/env bash
#
# Build and run the kiss_fftr pool / in-place / batch tests.
# No third-party dependency, no network access: only the repo's own
# sources and the system C compiler/libc are used.
#
# Runs two builds:
#   1. optimized build: correctness + allocation/RSS audit + benchmark
#   2. ASan/UBSan build: memory and undefined-behaviour check
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/artifacts/fftr_pool_build"
mkdir -p "$OUT"

SRC="$ROOT/test/test_fftr_pool.c"
WARN="-W -Wall -Wstrict-prototypes -Wmissing-prototypes -Wcast-align \
      -Wcast-qual -Wnested-externs -Wshadow -Wbad-function-cast -Wwrite-strings"

echo "################ 1. optimized build (-O2) ################"
# shellcheck disable=SC2086
cc -std=c99 -O2 $WARN -I"$ROOT" -o "$OUT/test_fftr_pool" $SRC -lm
"$OUT/test_fftr_pool"

echo
echo "################ 2. ASan + UBSan build ################"
# shellcheck disable=SC2086
cc -std=c99 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
   $WARN -I"$ROOT" -o "$OUT/test_fftr_pool_asan" $SRC -lm
ASAN_OPTIONS=detect_leaks=1 "$OUT/test_fftr_pool_asan"

echo
echo "################ 3. existing testsuite sanity (real FFT) ################"
# shellcheck disable=SC2086
cc -std=c99 -O2 $WARN -I"$ROOT" -o "$OUT/test_real" \
   "$ROOT/test/test_real.c" "$ROOT/kiss_fft.c" "$ROOT/kiss_fftr.c" -lm
"$OUT/test_real" 64

echo
echo "All runs completed. Binaries in $OUT"
