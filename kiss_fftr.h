/*
 *  Copyright (c) 2003-2004, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#ifndef KISS_FTR_H
#define KISS_FTR_H

#include "kiss_fft.h"
#ifdef __cplusplus
extern "C" {
#endif

    
/* 
 
 Real optimized version can save about 45% cpu time vs. complex fft of a real seq.

 
 
 */

typedef struct kiss_fftr_state *kiss_fftr_cfg;


kiss_fftr_cfg KISS_FFT_API kiss_fftr_alloc(int nfft,int inverse_fft,void * mem, size_t * lenmem);
/*
 nfft must be even

 If you don't care to allocate space, use mem = lenmem = NULL 
*/


void KISS_FFT_API kiss_fftr(kiss_fftr_cfg cfg,const kiss_fft_scalar *timedata,kiss_fft_cpx *freqdata);
/*
 input timedata has nfft scalar points
 output freqdata has nfft/2+1 complex points
*/

void KISS_FFT_API kiss_fftri(kiss_fftr_cfg cfg,const kiss_fft_cpx *freqdata,kiss_fft_scalar *timedata);
/*
 input freqdata has  nfft/2+1 complex points
 output timedata has nfft scalar points
 For floating-point builds, a kiss_fftr / kiss_fftri round trip returns
 nfft times the input; divide timedata by nfft to recover the input.
 Fixed-point builds scale internally and use a different convention.
*/

#define kiss_fftr_free KISS_FFT_FREE

/* ------------------------------------------------------------------ */
/* Reuse pool, in-place entries and batch processing.                 */
/*                                                                     */
/* A frame-oriented workload that runs hundreds of short real FFTs    */
/* no longer has to malloc/free a config object for every transform:  */
/* kiss_fftr_pool_acquire() hands out reusable kiss_fftr_cfg objects  */
/* and kiss_fftr_pool_release() returns them to the pool. All pooled  */
/* storage is released by kiss_fftr_pool_destroy().                   */
/*                                                                     */
/* These helpers are not thread-safe; use one pool per thread.        */
/* ------------------------------------------------------------------ */

typedef struct kiss_fftr_pool *kiss_fftr_pool_t;

/* Create/destroy a config reuse pool. The pool itself and every cfg
 * handed out by kiss_fftr_pool_acquire() are freed by _destroy();
 * configs must not be freed individually with kiss_fftr_free(). */
kiss_fftr_pool_t KISS_FFT_API kiss_fftr_pool_create(void);
void KISS_FFT_API kiss_fftr_pool_destroy(kiss_fftr_pool_t pool);

/* Get a forward (inverse_fft == 0) or inverse (inverse_fft != 0)
 * real-FFT config for nfft from the pool. A matching released config
 * is reused; otherwise a new one is allocated through kiss_fftr_alloc.
 * Returns NULL on invalid nfft or allocation failure. */
kiss_fftr_cfg KISS_FFT_API kiss_fftr_pool_acquire(kiss_fftr_pool_t pool,
                                                  int nfft,
                                                  int inverse_fft);

/* Return a config previously obtained from this pool. */
void KISS_FFT_API kiss_fftr_pool_release(kiss_fftr_pool_t pool,
                                         kiss_fftr_cfg cfg);

/* In-place forward real FFT.
 * buf holds nfft real input samples on entry and nfft/2+1 complex
 * output bins on return, so it must provide room for nfft+2 scalars
 * (i.e. (nfft/2+1) kiss_fft_cpx values). No memory is allocated. */
void KISS_FFT_API kiss_fftr_inplace(kiss_fftr_cfg cfg,
                                    kiss_fft_scalar *buf);

/* In-place inverse real FFT.
 * buf holds nfft/2+1 complex input bins on entry and nfft real output
 * samples on return, so it must provide room for nfft+2 scalars.
 * No memory is allocated. */
void KISS_FFT_API kiss_fftr_inplacei(kiss_fftr_cfg cfg,
                                     kiss_fft_scalar *buf);

/* One segment of a batch. nfft must be even and positive. */
typedef struct kiss_fftr_batch_seg {
    int nfft;
    kiss_fft_scalar *timedata; /* nfft real samples */
    kiss_fft_cpx *freqdata;    /* nfft/2+1 complex bins */
} kiss_fftr_batch_seg;

/* Run a forward real FFT on every segment. Segments of equal length
 * share one pooled config, so a batch of N equal-length segments costs
 * a single kiss_fftr_alloc instead of N. The grouping needs no extra
 * allocation. Returns the number of transforms performed. */
int KISS_FFT_API kiss_fftr_batch(kiss_fftr_pool_t pool,
                                 const kiss_fftr_batch_seg *segs,
                                 int nseg);

/* Inverse counterpart of kiss_fftr_batch. */
int KISS_FFT_API kiss_fftr_batchi(kiss_fftr_pool_t pool,
                                  const kiss_fftr_batch_seg *segs,
                                  int nseg);

#ifdef __cplusplus
}
#endif
#endif
