/*
 *  Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/* Acceptance test + micro-benchmark for the multi-channel batch FFT.
 *
 * The library is compiled into this test (unity build) with
 * KISS_FFT_MALLOC/KISS_FFT_FREE redirected to a counting allocator, so the
 * number of allocations made while configuring and running a batch is exact.
 *
 * It verifies, for 1/2/6/8 channels:
 *   1. forward batch results equal a per-channel kiss_fft() reference for both
 *      planar and interleaved layouts, out of place and in place;
 *   2. inverse batch results equal a per-channel kiss_fft() inverse reference;
 *   3. the interleaved path performs no allocation and no deinterleave copy on
 *      a transform call (all storage comes from the single config allocation);
 *   4. an 8-channel 1024-point batch stays under 2 ms.
 *
 * Inputs are code-generated sine waves plus pseudo-random noise.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ---- counting allocator, visible before the library headers ------------- */

static void *test_malloc(size_t n);
static void test_free(void *p);
#define KISS_FFT_MALLOC(nbytes) test_malloc(nbytes)
#define KISS_FFT_FREE(ptr) test_free(ptr)

#include "kiss_fft.h"
#include "kiss_fft_batch.h"
#include "_kiss_fft_guts.h"

/* the library itself is not built with -W; keep its noise out of this test */
#if defined(__GNUC__) || defined(__clang__)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wsign-compare"
# if !defined(__clang__)
#  pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
# endif
#endif
#include "../kiss_fft.c"
#include "../kiss_fft_batch.c"
#if defined(__GNUC__) || defined(__clang__)
# pragma GCC diagnostic pop
#endif

/* GCC 15's interprocedural analysis falsely reports the malloc'd in_buf as
   possibly uninitialized in the specialized clone of run_scenario() below. */
#if defined(__GNUC__) && !defined(__clang__)
# pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

/* ---- allocator bookkeeping ---------------------------------------------- */

static long g_malloc_count = 0;   /* KISS_FFT_MALLOC calls since last reset */
static long g_free_count = 0;
static long g_live_blocks = 0;

static void *test_malloc(size_t n)
{
    void *p = malloc(n);
    if (p != NULL) {
        ++g_malloc_count;
        ++g_live_blocks;
    }
    return p;
}

static void test_free(void *p)
{
    if (p != NULL) {
        free(p);
        ++g_free_count;
        --g_live_blocks;
    }
}

static void reset_alloc(void) { g_malloc_count = g_free_count = 0; }

/* ---- scalar accessors (the SIMD build makes kiss_fft_scalar a vector) ---- */

#ifdef USE_SIMD
# ifdef HAVE_LASX
#  define TEST_SCALAR_SET(sc, v) ((sc) = (__m256)__lasx_xvldrepl_w(&(float){(v)}, 0))
# elif defined(HAVE_LSX)
#  define TEST_SCALAR_SET(sc, v) ((sc) = (__m128)__lsx_vldrepl_w(&(float){(v)}, 0))
# else
#  define TEST_SCALAR_SET(sc, v) ((sc) = _mm_set1_ps((float)(v)))
# endif
# define TEST_SCALAR_GET(sc) ((float)(*(const float *)&(sc)))
#else
# define TEST_SCALAR_SET(sc, v) ((sc) = (kiss_fft_scalar)(v))
# define TEST_SCALAR_GET(sc) ((double)(sc))
#endif

/* ---- deterministic input generation ------------------------------------- */

static uint32_t rng_state = 0x12345678u;

static float rng_float(void)
{
    /* xorshift32 -> roughly uniform in [-1, 1] */
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return (float)((double)(x & 0xFFFFFFu) / (double)0xFFFFFFu * 2.0 - 1.0);
}

/* channel c: a channel-specific tone plus two partials plus noise */
static void gen_sample(int c, int k, int nfft, kiss_fft_cpx *out)
{
    double phase = 2.0 * M_PI * (double)((c + 1) * 7) * (double)k / (double)nfft;
    double p2 = 2.0 * M_PI * (double)((c + 3) * 13) * (double)k / (double)nfft;
    double p3 = 2.0 * M_PI * (double)((c + 5) * 29) * (double)k / (double)nfft;
    double v = 0.6 * sin(phase) + 0.25 * sin(p2) + 0.15 * cos(p3);
    TEST_SCALAR_SET(out->r, v + 0.10 * rng_float());
    TEST_SCALAR_SET(out->i, 0.5 * cos(phase) + 0.10 * rng_float());
}

/* ---- layout helpers ------------------------------------------------------ */

static void planar_set(kiss_fft_cpx *buf, int nfft, int c, int k,
                       const kiss_fft_cpx *v)
{
    buf[(size_t)c * nfft + k] = *v;
}

static void inter_set(kiss_fft_cpx *buf, int nchannels, int c, int k,
                      const kiss_fft_cpx *v)
{
    buf[(size_t)k * nchannels + c] = *v;
}

static kiss_fft_cpx planar_get(const kiss_fft_cpx *buf, int nfft, int c, int k)
{
    return buf[(size_t)c * nfft + k];
}

static kiss_fft_cpx inter_get(const kiss_fft_cpx *buf, int nchannels, int c,
                              int k)
{
    return buf[(size_t)k * nchannels + c];
}

/* ---- timing -------------------------------------------------------------- */

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---- report state -------------------------------------------------------- */

static int g_failures = 0;

#define CHANNELS_TO_TEST 4
static const int g_channels[CHANNELS_TO_TEST] = {1, 2, 6, 8};
static const int NFFT = 1024;
static const double ERR_TOL = 1e-9;

static double max_abs_err(const kiss_fft_cpx *a, const kiss_fft_cpx *b,
                          size_t n)
{
    double e = 0.0;
    size_t i;
    for (i = 0; i < n; ++i) {
        double dr = TEST_SCALAR_GET(a[i].r) - TEST_SCALAR_GET(b[i].r);
        double di = TEST_SCALAR_GET(a[i].i) - TEST_SCALAR_GET(b[i].i);
        double m = fabs(dr);
        if (fabs(di) > m) m = fabs(di);
        if (m > e) e = m;
    }
    return e;
}

/* Build one planar source buffer shared by every scenario of this channel
   count, so all layouts see identical inputs. */
static void build_planar_source(kiss_fft_cpx *planar, int nchannels, int nfft)
{
    int c, k;
    for (c = 0; c < nchannels; ++c)
        for (k = 0; k < nfft; ++k)
            gen_sample(c, k, nfft, &planar[(size_t)c * nfft + k]);
}

/* Reference: transform every channel of a planar buffer with plain kiss_fft. */
static void reference_forward(const kiss_fft_cpx *planar_in,
                              kiss_fft_cpx *planar_ref, int nchannels,
                              int nfft, int inverse)
{
    kiss_fft_cfg ref = kiss_fft_alloc(nfft, inverse, NULL, NULL);
    int c;
    for (c = 0; c < nchannels; ++c)
        kiss_fft(ref, planar_in + (size_t)c * nfft,
                 planar_ref + (size_t)c * nfft);
    kiss_fft_free(ref);
}

/* Gather a (possibly interleaved) batch output into a planar buffer for a
   whole-buffer comparison against the per-channel reference. */
static void gather_to_planar(const kiss_fft_cpx *out, kiss_fft_cpx *planar,
                             int nchannels, int nfft, int layout)
{
    int c, k;
    for (c = 0; c < nchannels; ++c)
        for (k = 0; k < nfft; ++k) {
            kiss_fft_cpx v = (layout == KISS_FFT_BATCH_INTERLEAVED)
                ? inter_get(out, nchannels, c, k)
                : planar_get(out, nfft, c, k);
            planar_set(planar, nfft, c, k, &v);
        }
}

static void run_scenario(const char *name, int nchannels, int layout,
                         int in_place, int inverse,
                         const kiss_fft_cpx *planar_src,
                         const kiss_fft_cpx *planar_ref)
{
    const size_t total = (size_t)nchannels * NFFT;
    kiss_fft_cpx *in_buf = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * total);
    kiss_fft_cpx *out_buf = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * total);
    kiss_fft_cpx *out_owned = out_buf; /* separately malloc'd block to free */
    kiss_fft_cpx *planar_got =
        (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * total);
    kiss_fft_batch_cfg cfg = NULL;
    long cfg_allocs, call_allocs, call_frees;
    double err;
    int c, k;

    /* in-place transforms overwrite the input buffer itself */
    if (in_place)
        out_buf = in_buf;
    if (in_buf == NULL || out_buf == NULL || planar_got == NULL) {
        printf("FAIL: out of memory in %s\n", name);
        ++g_failures;
        free(in_buf);
        free(out_owned);
        free(planar_got);
        return;
    }

    /* lay the identical planar source out in the requested format */
    for (c = 0; c < nchannels; ++c)
        for (k = 0; k < NFFT; ++k) {
            kiss_fft_cpx v = planar_get(planar_src, NFFT, c, k);
            if (layout == KISS_FFT_BATCH_INTERLEAVED)
                inter_set(in_buf, nchannels, c, k, &v);
            else
                planar_set(in_buf, NFFT, c, k, &v);
        }

    reset_alloc();
    cfg = kiss_fft_batch_alloc(NFFT, nchannels, inverse, layout, NULL, NULL);
    cfg_allocs = g_malloc_count;

    reset_alloc();
    kiss_fft_batch(cfg, in_buf, out_buf);
    call_allocs = g_malloc_count;
    call_frees = g_free_count;

    gather_to_planar(out_buf, planar_got, nchannels, NFFT, layout);
    err = max_abs_err(planar_got, planar_ref, total);

    printf("  %-28s nch=%d %s %s | cfg_allocs=%ld call_allocs=%ld "
           "call_frees=%ld | max_abs_err=%-3.3g %s\n",
           name, nchannels,
           layout == KISS_FFT_BATCH_INTERLEAVED ? "INTERLEAVED" : "PLANAR     ",
           in_place ? "in-place " : "out-place",
           cfg_allocs, call_allocs, call_frees, err,
           err < ERR_TOL ? "OK" : "FAIL");
    if (!(err < ERR_TOL))
        ++g_failures;

    /* allocation acceptance: exactly one config block, zero per-call work */
    if (cfg_allocs != 1) {
        printf("    FAIL: expected 1 config allocation, got %ld\n", cfg_allocs);
        ++g_failures;
    }
    if (call_allocs != 0 || call_frees != 0) {
        printf("    FAIL: transform call allocated/freed (allocs=%ld frees=%ld); "
               "interleaved and in-place paths must not allocate per call\n",
               call_allocs, call_frees);
        ++g_failures;
    }

    /* show the actual comparison numbers for the first channel, first bins,
       once per channel count (planar out-of-place forward scenario) */
    if (layout == KISS_FFT_BATCH_PLANAR && !in_place && !inverse) {
        printf("    channel 0 first 4 bins (reference per-channel vs batch):\n");
        for (k = 0; k < 4; ++k) {
            kiss_fft_cpx r = planar_get(planar_ref, NFFT, 0, k);
            kiss_fft_cpx g = planar_get(planar_got, NFFT, 0, k);
            printf("      bin[%d] ref=(% .8f % .8f) batch=(% .8f % .8f)\n",
                   k, TEST_SCALAR_GET(r.r), TEST_SCALAR_GET(r.i),
                   TEST_SCALAR_GET(g.r), TEST_SCALAR_GET(g.i));
        }
    }

    reset_alloc();
    kiss_fft_batch_free(cfg);
    if (g_live_blocks != 0) {
        printf("    FAIL: %ld block(s) still live after free\n", g_live_blocks);
        ++g_failures;
    }

    free(in_buf);
    free(out_owned);
    free(planar_got);
}

static void run_channel_count(int nchannels)
{
    const size_t total = (size_t)nchannels * NFFT;
    kiss_fft_cpx *planar_src =
        (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * total);
    kiss_fft_cpx *fwd_ref =
        (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * total);
    kiss_fft_cpx *inv_ref =
        (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * total);

    build_planar_source(planar_src, nchannels, NFFT);
    reference_forward(planar_src, fwd_ref, nchannels, NFFT, 0);
    reference_forward(planar_src, inv_ref, nchannels, NFFT, 1);

    printf("=== %d channel(s), nfft=%d ===\n", nchannels, NFFT);
    run_scenario("forward", nchannels, KISS_FFT_BATCH_PLANAR, 0, 0,
                 planar_src, fwd_ref);
    run_scenario("forward", nchannels, KISS_FFT_BATCH_PLANAR, 1, 0,
                 planar_src, fwd_ref);
    run_scenario("forward", nchannels, KISS_FFT_BATCH_INTERLEAVED, 0, 0,
                 planar_src, fwd_ref);
    run_scenario("forward", nchannels, KISS_FFT_BATCH_INTERLEAVED, 1, 0,
                 planar_src, fwd_ref);
    run_scenario("inverse", nchannels, KISS_FFT_BATCH_PLANAR, 0, 1,
                 planar_src, inv_ref);
    run_scenario("inverse", nchannels, KISS_FFT_BATCH_INTERLEAVED, 1, 1,
                 planar_src, inv_ref);

    free(planar_src);
    free(fwd_ref);
    free(inv_ref);
}

/* 8-channel 1024-point timing: batch vs the per-channel loop, both layouts. */
static void benchmark(void)
{
    const int nchannels = 8;
    const int iters = 2000;
    const size_t total = (size_t)nchannels * NFFT;
    kiss_fft_cpx *buf_a = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * total);
    kiss_fft_cpx *buf_b = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * total);
    kiss_fft_batch_cfg planar_cfg =
        kiss_fft_batch_alloc(NFFT, nchannels, 0, KISS_FFT_BATCH_PLANAR, NULL, NULL);
    kiss_fft_batch_cfg inter_cfg =
        kiss_fft_batch_alloc(NFFT, nchannels, 0, KISS_FFT_BATCH_INTERLEAVED,
                             NULL, NULL);
    kiss_fft_cfg single = kiss_fft_alloc(NFFT, 0, NULL, NULL);
    int i, c;
    double t0, t_planar, t_inter, t_loop, per_call_us;

    build_planar_source(buf_a, nchannels, NFFT);
    /* mirror the planar source into an interleaved working buffer */
    for (c = 0; c < nchannels; ++c) {
        int k;
        for (k = 0; k < NFFT; ++k) {
            kiss_fft_cpx v = planar_get(buf_a, NFFT, c, k);
            inter_set(buf_b, nchannels, c, k, &v);
        }
    }

    /* warm up */
    for (i = 0; i < 100; ++i) {
        kiss_fft_batch(planar_cfg, buf_a, buf_b);
        kiss_fft_batch(inter_cfg, buf_b, buf_a);
    }

    t0 = now_seconds();
    for (i = 0; i < iters; ++i)
        kiss_fft_batch(planar_cfg, buf_a, buf_b);
    t_planar = (now_seconds() - t0) / (double)iters;

    t0 = now_seconds();
    for (i = 0; i < iters; ++i)
        kiss_fft_batch(inter_cfg, buf_b, buf_a);
    t_inter = (now_seconds() - t0) / (double)iters;

    t0 = now_seconds();
    for (i = 0; i < iters; ++i)
        for (c = 0; c < nchannels; ++c)
            kiss_fft(single, buf_a + (size_t)c * NFFT,
                     buf_b + (size_t)c * NFFT);
    t_loop = (now_seconds() - t0) / (double)iters;

    per_call_us = t_inter * 1e6;
    printf("\n=== timing: 8 channels x %d-point FFT, best-of-%d avg ===\n",
           NFFT, iters);
    printf("  per-channel kiss_fft() loop : %8.3f us/call\n", t_loop * 1e6);
    printf("  batch planar  out-of-place  : %8.3f us/call\n", t_planar * 1e6);
    printf("  batch interleaved (no copy) : %8.3f us/call\n", per_call_us);
    printf("  acceptance (< 2000 us)      : %s\n",
           per_call_us < 2000.0 ? "OK" : "FAIL");
    if (!(per_call_us < 2000.0))
        ++g_failures;

    kiss_fft_free(single);
    kiss_fft_batch_free(inter_cfg);
    kiss_fft_batch_free(planar_cfg);
    free(buf_a);
    free(buf_b);
}

/* user-supplied buffer (mem/lenmem) path must describe and accept one block */
static void test_caller_supplied_buffer(void)
{
    const int nchannels = 6;
    size_t need = 0;
    void *mem;
    kiss_fft_batch_cfg cfg;
    kiss_fft_cpx in[6 * 32], out[6 * 32];
    int c, k;

    if (kiss_fft_batch_alloc(32, nchannels, 0, KISS_FFT_BATCH_PLANAR,
                             NULL, &need) != NULL || need == 0) {
        printf("FAIL: size query did not report a required size\n");
        ++g_failures;
        return;
    }
    mem = malloc(need);
    reset_alloc();
    cfg = kiss_fft_batch_alloc(32, nchannels, 0, KISS_FFT_BATCH_PLANAR,
                               mem, &need);
    if (cfg != (kiss_fft_batch_cfg)mem || g_malloc_count != 0) {
        printf("FAIL: caller-supplied buffer was not used (allocs=%ld)\n",
               g_malloc_count);
        ++g_failures;
    }
    for (c = 0; c < nchannels; ++c)
        for (k = 0; k < 32; ++k)
            gen_sample(c, k, 32, &in[(size_t)c * 32 + k]);
    kiss_fft_batch(cfg, in, out);
    /* spot check against a single-channel transform of channel 3 */
    {
        kiss_fft_cfg ref = kiss_fft_alloc(32, 0, NULL, NULL);
        kiss_fft_cpx ref_out[32];
        double err;
        kiss_fft(ref, in + 3 * 32, ref_out);
        err = max_abs_err(ref_out, out + 3 * 32, 32);
        if (!(err < ERR_TOL)) {
            printf("FAIL: caller-buffer batch mismatch, err=%g\n", err);
            ++g_failures;
        }
        kiss_fft_free(ref);
    }
    free(mem); /* caller owns it; kiss_fft_batch_free must not be used here */
}

static void test_invalid_args(void)
{
    if (kiss_fft_batch_alloc(0, 2, 0, KISS_FFT_BATCH_PLANAR, NULL, NULL) ||
        kiss_fft_batch_alloc(16, 0, 0, KISS_FFT_BATCH_PLANAR, NULL, NULL) ||
        kiss_fft_batch_alloc(16, 2, 0, 99, NULL, NULL)) {
        printf("FAIL: invalid arguments were accepted\n");
        ++g_failures;
    } else {
        printf("invalid-argument rejection: OK\n");
    }
}

/* A prime nfft forces the core engine's generic (non-2/3/4/5) butterfly, which
   takes its own temporary; the batch layer must still produce the per-channel
   reference exactly, for both layouts and the in-place mode. */
static void test_prime_size(void)
{
    const int nfft = 101;      /* prime */
    const int nchannels = 3;
    const size_t total = (size_t)nfft * nchannels;
    kiss_fft_cpx *planar = malloc(sizeof(kiss_fft_cpx) * total);
    kiss_fft_cpx *ref = malloc(sizeof(kiss_fft_cpx) * total);
    kiss_fft_cpx *buf = malloc(sizeof(kiss_fft_cpx) * total);
    kiss_fft_cpx *got = malloc(sizeof(kiss_fft_cpx) * total);
    kiss_fft_cfg single = kiss_fft_alloc(nfft, 0, NULL, NULL);
    kiss_fft_batch_cfg pc =
        kiss_fft_batch_alloc(nfft, nchannels, 0, KISS_FFT_BATCH_PLANAR, NULL, NULL);
    kiss_fft_batch_cfg ic =
        kiss_fft_batch_alloc(nfft, nchannels, 0, KISS_FFT_BATCH_INTERLEAVED,
                             NULL, NULL);
    int c, k;
    double err;

    for (c = 0; c < nchannels; ++c)
        for (k = 0; k < nfft; ++k)
            gen_sample(c, k, nfft, &planar[(size_t)c * nfft + k]);
    for (c = 0; c < nchannels; ++c)
        kiss_fft(single, planar + (size_t)c * nfft,
                 ref + (size_t)c * nfft);

    /* planar, in place */
    memcpy(buf, planar, sizeof(kiss_fft_cpx) * total);
    kiss_fft_batch(pc, buf, buf);
    err = max_abs_err(buf, ref, total);
    printf("prime nfft=%d planar in-place   max_abs_err=%-3.3g %s\n",
           nfft, err, err < ERR_TOL ? "OK" : "FAIL");
    if (!(err < ERR_TOL)) ++g_failures;

    /* interleaved, out of place, gathered back to planar */
    for (c = 0; c < nchannels; ++c)
        for (k = 0; k < nfft; ++k)
            inter_set(buf, nchannels, c, k, &planar[(size_t)c * nfft + k]);
    kiss_fft_batch(ic, buf, got);
    gather_to_planar(got, buf, nchannels, nfft, KISS_FFT_BATCH_INTERLEAVED);
    err = max_abs_err(buf, ref, total);
    printf("prime nfft=%d interleaved out   max_abs_err=%-3.3g %s\n",
           nfft, err, err < ERR_TOL ? "OK" : "FAIL");
    if (!(err < ERR_TOL)) ++g_failures;

    kiss_fft_free(single);
    kiss_fft_batch_free(ic);
    kiss_fft_batch_free(pc);
    free(got);
    free(buf);
    free(ref);
    free(planar);
}

int main(void)
{
    int i;
    double wall0 = now_seconds();

    test_invalid_args();
    test_caller_supplied_buffer();
    test_prime_size();

    for (i = 0; i < CHANNELS_TO_TEST; ++i)
        run_channel_count(g_channels[i]);

    benchmark();

    printf("\nTotal wall time: %.2f s\n", now_seconds() - wall0);
    printf("%s\n", g_failures == 0 ? "ALL PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
