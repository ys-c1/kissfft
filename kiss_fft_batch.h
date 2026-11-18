/*
 *  Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

/*
 * Multi-channel batch complex FFTs.
 *
 * A single kiss_fft_batch_cfg stores the transform size, the channel count,
 * the in-memory channel layout and the per-transform working storage, so that
 * one call transforms N channels sharing the same configuration. This replaces
 * the hand-written loop of N kiss_fft() calls and keeps every per-call
 * allocation out of the hot path: all storage is reserved once at alloc time.
 *
 * This module is a thin layer on top of the core kiss_fft transform. It never
 * touches the radix-2/radix-4 butterfly numerics; each channel is computed by
 * the same kf_work() engine a single-channel kiss_fft() call uses.
 */

#ifndef KISS_FFT_BATCH_H
#define KISS_FFT_BATCH_H

#include "kiss_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Channel layout of the caller's buffers. Both layouts hold exactly
 * nchannels * nfft contiguous kiss_fft_cpx values.
 *
 * KISS_FFT_BATCH_PLANAR ("deinterleaved" / channel-separated):
 *   channel c, sample k is stored at base[c * nfft + k].
 *   Each channel occupies one contiguous nfft-element block.
 *
 * KISS_FFT_BATCH_INTERLEAVED ("interwoven"):
 *   channel c, sample k is stored at base[k * nchannels + c].
 *   The channels are interleaved sample by sample; the per-channel element
 *   stride is nchannels.
 */
#define KISS_FFT_BATCH_PLANAR 0
#define KISS_FFT_BATCH_INTERLEAVED 1

typedef struct kiss_fft_batch_state *kiss_fft_batch_cfg;

/*
 *  kiss_fft_batch_alloc
 *
 *  Allocate the configuration and all working storage for transforming
 *  nchannels channels of nfft-point complex FFTs (or IFFTs, when
 *  inverse_fft is nonzero) with the given layout.
 *
 *  The mem / lenmem pair follows the same contract as kiss_fft_alloc():
 *    - lenmem == NULL: storage is allocated internally and must later be
 *      released with kiss_fft_batch_free().
 *    - lenmem != NULL, mem != NULL and *lenmem large enough: the state is
 *      placed in the caller's buffer and *lenmem receives the bytes used.
 *    - lenmem != NULL otherwise: NULL is returned and *lenmem receives the
 *      minimum required buffer size.
 *
 *  Returns NULL for an invalid nfft, nchannels or layout. The block returned
 *  (or described in *lenmem) is one contiguous allocation that embeds the
 *  underlying kiss_fft_cfg and a one-channel scratch buffer. The batch layer
 *  itself reserves every buffer it needs here, so kiss_fft_batch() performs no
 *  allocation of its own. (As with a plain kiss_fft(), an nfft whose factors
 *  exceed 2/3/4/5 still makes the core engine take its own generic-butterfly
 *  temporary; the common 2/3/4/5-smooth sizes, e.g. 1024, never do.)
 */
kiss_fft_batch_cfg KISS_FFT_API
kiss_fft_batch_alloc(int nfft, int nchannels, int inverse_fft, int layout,
                     void *mem, size_t *lenmem);

/*
 *  kiss_fft_batch
 *
 *  Transform every channel of fin and write the result to fout.
 *
 *  fin and fout each point to nchannels * nfft contiguous kiss_fft_cpx values
 *  arranged according to the layout given at alloc time. fin and fout may be
 *  the same pointer to request an in-place transform; the internally reserved
 *  scratch buffer is then used, so the batch layer itself allocates nothing on
 *  a call.
 *
 *  Layout behaviour and extra per-call work:
 *    - planar, out of place : each channel is transformed straight into its
 *      output block. No allocation and no extra copy.
 *    - interleaved input    : the channel samples are read directly with an
 *      element stride of nchannels; the input is never deinterleaved into a
 *      temporary planar buffer.
 *    - interleaved output   : the contiguous transform result is scattered
 *      into the interleaved output from the one-channel scratch reserved at
 *      alloc time (the core engine writes contiguous output only).
 *    - in place             : the reserved scratch backs the transform and,
 *      for interleaved buffers, the write-back.
 *
 *  For a 2/3/4/5-smooth nfft the whole call is allocation free; a prime-sized
 *  nfft lets the core engine allocate its generic-butterfly temporary exactly
 *  as a single-channel kiss_fft() would.
 *
 *  Each channel's result is bit-for-bit the output of a kiss_fft() call on
 *  that channel alone, because both run the same engine on the same input.
 */
void KISS_FFT_API
kiss_fft_batch(kiss_fft_batch_cfg cfg, const kiss_fft_cpx *fin,
               kiss_fft_cpx *fout);

/* If kiss_fft_batch_alloc allocated the state, it is one contiguous buffer and
   can be released with the library allocator. No-op for a caller-owned buffer. */
#define kiss_fft_batch_free KISS_FFT_FREE

#ifdef __cplusplus
}
#endif

#endif /* KISS_FFT_BATCH_H */
