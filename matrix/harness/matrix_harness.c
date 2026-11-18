/*
 * matrix_harness.c - single source used by every cell of the build-feature matrix.
 *
 * It is compiled once per CMake configuration (float / double / int16_t /
 * int32_t / simd, sequential / OpenMP) and linked against the matching static
 * libkissfft-*.a, so kiss_fft_scalar and every code path come from the library
 * it is paired with.
 *
 * Usage:
 *   matrix_harness <mode> <nfft> <in.bin> <out.bin>
 *     mode = cpx   : complex forward FFT,  N complex doubles in, N cpx out
 *     mode = real  : real forward FFT,     N complex doubles in (imag ignored),
 *                    N/2+1 cpx out
 *     mode = rt    : real forward+inverse round trip, N real samples out
 *
 * Input  file: native-endian float64 complex pairs (2*N doubles).
 * Output file: one text line per produced element, doubles in a fixed unit
 *              scale (see MATRIX_NOTES below), so the Python side never has to
 *              know the underlying scalar type.
 *
 * Fixed-point scaling convention (from _kiss_fft_guts.h C_FIXDIV +
 * kiss_fftr.c): every radix-p butterfly divides by p, hence a length-N
 * transform returns DFT/N. Reconstruction therefore multiplies by N here.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "kiss_fft.h"
#include "kiss_fftr.h"

/* ---- portability shims ------------------------------------------------- */

#if defined(USE_SIMD)
#  if defined(HAVE_LASX) || defined(HAVE_LSX)
#    define MAT_ALLOC(n)  aligned_alloc(16, (((n) + 15UL) & ~15UL))
#    define MAT_FREE(p)   free(p)
#  else
#    include <xmmintrin.h>
#    define MAT_ALLOC(n)  _mm_malloc((n), 16)
#    define MAT_FREE(p)   _mm_free(p)
#  endif
#else
#  define MAT_ALLOC(n)  malloc(n)
#  define MAT_FREE(p)   free(p)
#endif

#if defined(FIXED_POINT)
#  include <limits.h>
#  if FIXED_POINT == 32
#    define MAT_SAMP_MAX INT32_MAX
#    define MAT_SAMP_MIN INT32_MIN
#  else
#    define MAT_SAMP_MAX INT16_MAX
#    define MAT_SAMP_MIN INT16_MIN
#  endif
#endif

/* ---- convert one float64 sample into the build's scalar ---------------- */

static kiss_fft_scalar to_scalar(double x)
{
#if defined(USE_SIMD)
#  if defined(HAVE_LASX)
    float v = (float)x;
    return (__m256)__lasx_xvldrepl_w(&v, 0);
#  elif defined(HAVE_LSX)
    float v = (float)x;
    return (__m128)__lsx_vldrepl_w(&v, 0);
#  else
    return _mm_set1_ps((float)x);
#  endif
#elif defined(FIXED_POINT)
    double r = x * (double)MAT_SAMP_MAX;
    if (r > (double)MAT_SAMP_MAX) r = (double)MAT_SAMP_MAX;
    if (r < (double)MAT_SAMP_MIN) r = (double)MAT_SAMP_MIN;
    return (kiss_fft_scalar)lround(r);
#else
    return (kiss_fft_scalar)x;
#endif
}

/* ---- convert one output scalar back to float64 ------------------------- */

static double from_scalar(kiss_fft_scalar x)
{
#if defined(USE_SIMD)
    return (double)(*(float *)&x);
#elif defined(FIXED_POINT)
    return (double)x / (double)MAT_SAMP_MAX;
#else
    return (double)x;
#endif
}

static double *load_input(const char *path, int nfft)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen input"); exit(2); }
    size_t want = (size_t)nfft * 2u * sizeof(double);
    double *buf = (double *)malloc(want);
    if (!buf) { fputs("oom\n", stderr); exit(2); }
    size_t got = fread(buf, 1, want, f);
    fclose(f);
    if (got != want) {
        fprintf(stderr, "short input: got %zu want %zu\n", got, want);
        exit(2);
    }
    return buf;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "usage: %s cpx|real|rt <nfft> <in.bin> <out.bin>\n",
                argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    int nfft = atoi(argv[2]);
    if (nfft <= 0) {
        fprintf(stderr, "bad nfft\n");
        return 2;
    }

    double *in = load_input(argv[3], nfft);
    FILE *of = fopen(argv[4], "wb");
    if (!of) { perror("fopen output"); return 2; }

    if (strcmp(mode, "cpx") == 0) {
        kiss_fft_cpx *fin =
            (kiss_fft_cpx *)MAT_ALLOC(sizeof(kiss_fft_cpx) * nfft);
        kiss_fft_cpx *fout =
            (kiss_fft_cpx *)MAT_ALLOC(sizeof(kiss_fft_cpx) * nfft);
        if (!fin || !fout) { fputs("oom\n", stderr); return 2; }

        for (int i = 0; i < nfft; ++i) {
            fin[i].r = to_scalar(in[2 * i]);
            fin[i].i = to_scalar(in[2 * i + 1]);
        }

        kiss_fft_cfg cfg = kiss_fft_alloc(nfft, 0, NULL, NULL);
        if (!cfg) { fputs("kiss_fft_alloc failed\n", stderr); return 2; }
        kiss_fft(cfg, fin, fout);

        /* fixed point output is DFT/N; multiply by N to reach DFT scale */
        double g =
#if defined(FIXED_POINT)
            (double)nfft;
#else
            1.0;
#endif
        for (int i = 0; i < nfft; ++i)
            fprintf(of, "%.17g %.17g\n",
                    g * from_scalar(fout[i].r),
                    g * from_scalar(fout[i].i));

        kiss_fft_free(cfg);
        MAT_FREE(fin);
        MAT_FREE(fout);
    } else if (strcmp(mode, "real") == 0) {
        if (nfft & 1) { fputs("real mode needs even nfft\n", stderr); return 2; }
        int nb = nfft / 2 + 1;
        kiss_fft_scalar *timed =
            (kiss_fft_scalar *)MAT_ALLOC(sizeof(kiss_fft_scalar) * nfft);
        kiss_fft_cpx *freq =
            (kiss_fft_cpx *)MAT_ALLOC(sizeof(kiss_fft_cpx) * nb);
        if (!timed || !freq) { fputs("oom\n", stderr); return 2; }

        for (int i = 0; i < nfft; ++i)
            timed[i] = to_scalar(in[2 * i]); /* imaginary part discarded */

        kiss_fftr_cfg cfg = kiss_fftr_alloc(nfft, 0, NULL, NULL);
        if (!cfg) { fputs("kiss_fftr_alloc failed\n", stderr); return 2; }
        kiss_fftr(cfg, timed, freq);

        double g =
#if defined(FIXED_POINT)
            (double)nfft;
#else
            1.0;
#endif
        for (int i = 0; i < nb; ++i)
            fprintf(of, "%.17g %.17g\n",
                    g * from_scalar(freq[i].r),
                    g * from_scalar(freq[i].i));

        kiss_fftr_free(cfg);
        MAT_FREE(timed);
        MAT_FREE(freq);
    } else if (strcmp(mode, "rt") == 0) {
        if (nfft & 1) { fputs("rt mode needs even nfft\n", stderr); return 2; }
        int nb = nfft / 2 + 1;
        kiss_fft_scalar *timed =
            (kiss_fft_scalar *)MAT_ALLOC(sizeof(kiss_fft_scalar) * nfft);
        kiss_fft_scalar *back =
            (kiss_fft_scalar *)MAT_ALLOC(sizeof(kiss_fft_scalar) * nfft);
        kiss_fft_cpx *freq =
            (kiss_fft_cpx *)MAT_ALLOC(sizeof(kiss_fft_cpx) * nb);
        if (!timed || !back || !freq) { fputs("oom\n", stderr); return 2; }

        for (int i = 0; i < nfft; ++i)
            timed[i] = to_scalar(in[2 * i]);

        kiss_fftr_cfg cf = kiss_fftr_alloc(nfft, 0, NULL, NULL);
        kiss_fftr_cfg ci = kiss_fftr_alloc(nfft, 1, NULL, NULL);
        if (!cf || !ci) { fputs("kiss_fftr_alloc failed\n", stderr); return 2; }
        kiss_fftr(cf, timed, freq);
        kiss_fftri(ci, freq, back);

        /*
         * floating point: fftri returns nfft * input.
         * fixed point: forward fftr scales by 1/nfft and inverse fftri by
         *              1/nfft again, so the round trip comes back as
         *              input / nfft; multiply by nfft.
         */
#if defined(FIXED_POINT)
        double g = (double)nfft;
#else
        double g = 1.0 / (double)nfft;
#endif
        for (int i = 0; i < nfft; ++i)
            fprintf(of, "%.17g\n", g * from_scalar(back[i]));

        kiss_fftr_free(cf);
        kiss_fftr_free(ci);
        MAT_FREE(timed);
        MAT_FREE(back);
        MAT_FREE(freq);
    } else {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    fclose(of);
    free(in);
    return 0;
}
