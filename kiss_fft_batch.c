/*
 *  Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/*
 * Multi-channel batch complex FFT.
 *
 * The whole configuration is one contiguous block that packs three objects:
 *
 *     struct kiss_fft_batch_state   (the state below)
 *     struct kiss_fft_state         (the single-channel engine configuration,
 *                                    shared by every channel of the batch)
 *     kiss_fft_cpx scratch[nfft]    (one-channel working buffer)
 *
 * The single-channel state is reused for every channel, and the scratch is
 * reserved here so that interleaved output and in-place transforms allocate
 * nothing during kiss_fft_batch(). The radix-2/3/4/5 butterflies and kf_work()
 * are used unchanged; this file only orchestrates per-channel calls and the
 * two memory layouts.
 */

#include "kiss_fft_batch.h"
#include "_kiss_fft_guts.h"
#include <stdint.h>

struct kiss_fft_batch_state {
    int nfft;
    int nchannels;
    int inverse;
    int layout;                 /* KISS_FFT_BATCH_PLANAR / _INTERLEAVED */
    kiss_fft_cfg subcfg;        /* embedded single-channel configuration */
    kiss_fft_cpx *scratch;      /* one nfft-point working buffer */
};

kiss_fft_batch_cfg
kiss_fft_batch_alloc(int nfft, int nchannels, int inverse_fft, int layout,
                     void *mem, size_t *lenmem)
{
    KISS_FFT_ALIGN_CHECK(mem)

    kiss_fft_batch_cfg st = NULL;
    size_t sublen = 0;
    size_t scratch_bytes;
    size_t memneeded;
    char *ptr = NULL;

    /* reject bad transform sizes, bad channel counts and unknown layouts */
    if (nfft <= 0 || nchannels <= 0 ||
        (layout != KISS_FFT_BATCH_PLANAR &&
         layout != KISS_FFT_BATCH_INTERLEAVED))
        return NULL;
    /* mirror the overflow guard in kiss_fft_alloc before any size arithmetic */
    if ((size_t)nfft >=
        (SIZE_MAX - 2 * sizeof(struct kiss_fft_state)) / sizeof(kiss_fft_cpx))
        return NULL;

    /* size the embedded single-channel configuration; this only reports a
       size, it does not allocate (mem == NULL, lenmem != &sublen) */
    (void)kiss_fft_alloc(nfft, inverse_fft, NULL, &sublen);
    if (sublen == 0)
        return NULL;

    scratch_bytes = sizeof(kiss_fft_cpx) * (size_t)nfft;
    memneeded = KISS_FFT_PACK_SIZE_UP(sizeof(struct kiss_fft_batch_state))
              + KISS_FFT_PACK_SIZE_UP(sublen)
              + KISS_FFT_PACK_SIZE_UP(scratch_bytes);

    if (lenmem == NULL) {
        ptr = (char *)KISS_FFT_MALLOC(memneeded);
    } else {
        if (mem != NULL && *lenmem >= memneeded)
            ptr = (char *)mem;
        *lenmem = memneeded;
    }
    if (!ptr)
        return NULL;

    st = (kiss_fft_batch_cfg)ptr;
    ptr += KISS_FFT_PACK_SIZE_UP(sizeof(struct kiss_fft_batch_state));

    st->subcfg = (kiss_fft_cfg)ptr;
    ptr += KISS_FFT_PACK_SIZE_UP(sublen);

    st->scratch = (kiss_fft_cpx *)ptr;
    ptr += KISS_FFT_PACK_SIZE_UP(scratch_bytes);

    st->nfft = nfft;
    st->nchannels = nchannels;
    st->inverse = inverse_fft;
    st->layout = layout;

    /* build the engine state inside the packed block */
    (void)kiss_fft_alloc(nfft, inverse_fft, st->subcfg, &sublen);

    return st;
}

void
kiss_fft_batch(kiss_fft_batch_cfg st, const kiss_fft_cpx *fin,
               kiss_fft_cpx *fout)
{
    int c, k;
    const int nfft = st->nfft;
    const int nchannels = st->nchannels;
    const int interleaved = (st->layout == KISS_FFT_BATCH_INTERLEAVED);
    const int in_place = (fin == fout);

    if (fin == NULL || fout == NULL) {
        KISS_FFT_ERROR("kiss_fft_batch: NULL buffer.");
        return;
    }

    /* The core engine only emits contiguous output and allocates a temp block
       itself when handed fin == fout. Both cases are served by the scratch
       reserved at alloc time, so this routine never allocates per call.
       Planar, out-of-place channels need neither scratch nor extra copy. */
    const int use_scratch = in_place || interleaved;

    for (c = 0; c < nchannels; ++c) {
        const kiss_fft_cpx *in;
        int in_stride;

        if (interleaved) {
            /* channel c is column c of the [nfft x nchannels] layout: read it
               directly, no deinterleave copy into a planar buffer */
            in = fin + c;
            in_stride = nchannels;
        } else {
            in = fin + (size_t)c * nfft;
            in_stride = 1;
        }

        if (!use_scratch) {
            /* planar, out of place: transform straight into the output block */
            kiss_fft_stride(st->subcfg, in, fout + (size_t)c * nfft,
                            in_stride);
            continue;
        }

        /* contiguous transform result lands in the reserved scratch */
        kiss_fft_stride(st->subcfg, in, st->scratch, in_stride);

        if (interleaved) {
            /* scatter the contiguous result into output column c; this only
               ever touches column c, so it cannot clobber another channel's
               input, even when fin == fout */
            for (k = 0; k < nfft; ++k)
                fout[(size_t)k * nchannels + c] = st->scratch[k];
        } else {
            /* planar in place: put the channel back into its own block */
            memcpy(fout + (size_t)c * nfft, st->scratch,
                   sizeof(kiss_fft_cpx) * (size_t)nfft);
        }
    }
}
