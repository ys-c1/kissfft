/*
 *  Copyright (c) 2003-2004, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#include "kiss_fftr.h"
#include "_kiss_fft_guts.h"

struct kiss_fftr_state{
    kiss_fft_cfg substate;
    kiss_fft_cpx * tmpbuf;
    kiss_fft_cpx * super_twiddles;
#ifdef USE_SIMD
    void * pad;
#endif
};

kiss_fftr_cfg kiss_fftr_alloc(int nfft,int inverse_fft,void * mem,size_t * lenmem)
{
	KISS_FFT_ALIGN_CHECK(mem)

    int i;
    kiss_fftr_cfg st = NULL;
    size_t subsize = 0, memneeded;

    if (nfft <= 0 || (nfft & 1)) {
        KISS_FFT_ERROR("Real FFT optimization must be even and positive.");
        return NULL;
    }
    nfft >>= 1;

    /* size query only: mem == NULL with lenmem set makes kiss_fft_alloc
     * report the required bytes without allocating */
    kiss_fft_alloc (nfft, inverse_fft, NULL, &subsize);
    memneeded = sizeof(struct kiss_fftr_state) + subsize + sizeof(kiss_fft_cpx) * ( (size_t)nfft * 3 / 2);

    if (lenmem == NULL) {
        st = (kiss_fftr_cfg) KISS_FFT_MALLOC (memneeded);
    } else {
        if (*lenmem >= memneeded)
            st = (kiss_fftr_cfg) mem;
        *lenmem = memneeded;
    }
    if (!st)
        return NULL;

    st->substate = (kiss_fft_cfg) (st + 1); /*just beyond kiss_fftr_state struct */
    st->tmpbuf = (kiss_fft_cpx *) (((char *) st->substate) + subsize);
    st->super_twiddles = st->tmpbuf + nfft;
    kiss_fft_alloc(nfft, inverse_fft, st->substate, &subsize);

    for (i = 0; i < nfft/2; ++i) {
        double phase =
            -3.14159265358979323846264338327 * ((double) (i+1) / nfft + .5);
        if (inverse_fft)
            phase *= -1;
        kf_cexp (st->super_twiddles+i,phase);
    }
    return st;
}

void kiss_fftr(kiss_fftr_cfg st,const kiss_fft_scalar *timedata,kiss_fft_cpx *freqdata)
{
    /* input buffer timedata is stored row-wise */
    int k,ncfft;
    kiss_fft_cpx fpnk,fpk,f1k,f2k,tw,tdc;

    if ( st->substate->inverse) {
        KISS_FFT_ERROR("kiss fft usage error: improper alloc");
        return;/* The caller did not call the correct function */
    }

    ncfft = st->substate->nfft;

    /*perform the parallel fft of two real signals packed in real,imag*/
    kiss_fft( st->substate , (const kiss_fft_cpx*)timedata, st->tmpbuf );
    /* The real part of the DC element of the frequency spectrum in st->tmpbuf
     * contains the sum of the even-numbered elements of the input time sequence
     * The imag part is the sum of the odd-numbered elements
     *
     * The sum of tdc.r and tdc.i is the sum of the input time sequence.
     *      yielding DC of input time sequence
     * The difference of tdc.r - tdc.i is the sum of the input (dot product) [1,-1,1,-1...
     *      yielding Nyquist bin of input time sequence
     */

    tdc.r = st->tmpbuf[0].r;
    tdc.i = st->tmpbuf[0].i;
    C_FIXDIV(tdc,2);
    CHECK_OVERFLOW_OP(tdc.r ,+, tdc.i);
    CHECK_OVERFLOW_OP(tdc.r ,-, tdc.i);
    freqdata[0].r = tdc.r + tdc.i;
    freqdata[ncfft].r = tdc.r - tdc.i;
#ifdef USE_SIMD
#ifdef HAVE_LASX
    freqdata[0].i = (__m256)(__lasx_xvreplgr2vr_w(0));
    freqdata[ncfft].i = freqdata[0].i;
#elif defined(HAVE_LSX)
    freqdata[0].i = (__m128)(__lsx_vreplgr2vr_w(0));
    freqdata[ncfft].i = freqdata[0].i;
#else
    freqdata[ncfft].i = freqdata[0].i = _mm_set1_ps(0);
#endif
#else
    freqdata[ncfft].i = freqdata[0].i = 0;
#endif

    for ( k=1;k <= ncfft/2 ; ++k ) {
        fpk    = st->tmpbuf[k];
        fpnk.r =   st->tmpbuf[ncfft-k].r;
        fpnk.i = - st->tmpbuf[ncfft-k].i;
        C_FIXDIV(fpk,2);
        C_FIXDIV(fpnk,2);

        C_ADD( f1k, fpk , fpnk );
        C_SUB( f2k, fpk , fpnk );
        C_MUL( tw , f2k , st->super_twiddles[k-1]);

        freqdata[k].r = HALF_OF(f1k.r + tw.r);
        freqdata[k].i = HALF_OF(f1k.i + tw.i);
        freqdata[ncfft-k].r = HALF_OF(f1k.r - tw.r);
        freqdata[ncfft-k].i = HALF_OF(tw.i - f1k.i);
    }
}

void kiss_fftri(kiss_fftr_cfg st,const kiss_fft_cpx *freqdata,kiss_fft_scalar *timedata)
{
    /* input buffer timedata is stored row-wise */
    int k, ncfft;

    if (st->substate->inverse == 0) {
        KISS_FFT_ERROR("kiss fft usage error: improper alloc");
        return;/* The caller did not call the correct function */
    }

    ncfft = st->substate->nfft;

    st->tmpbuf[0].r = freqdata[0].r + freqdata[ncfft].r;
    st->tmpbuf[0].i = freqdata[0].r - freqdata[ncfft].r;
    C_FIXDIV(st->tmpbuf[0],2);

    for (k = 1; k <= ncfft / 2; ++k) {
        kiss_fft_cpx fk, fnkc, fek, fok, tmp;
        fk = freqdata[k];
        fnkc.r = freqdata[ncfft - k].r;
        fnkc.i = -freqdata[ncfft - k].i;
        C_FIXDIV( fk , 2 );
        C_FIXDIV( fnkc , 2 );

        C_ADD (fek, fk, fnkc);
        C_SUB (tmp, fk, fnkc);
        C_MUL (fok, tmp, st->super_twiddles[k-1]);
        C_ADD (st->tmpbuf[k],     fek, fok);
        C_SUB (st->tmpbuf[ncfft - k], fek, fok);
#ifdef USE_SIMD
#ifdef HAVE_LASX
        __m256 neg_one = (__m256)__lasx_xvreplgr2vr_w(0xBF800000); // -1.0f
        st->tmpbuf[ncfft - k].i = __lasx_xvfmul_s(st->tmpbuf[ncfft - k].i, neg_one);
#elif defined(HAVE_LSX)
        __m128 neg_one = (__m128)__lsx_vreplgr2vr_w(0xBF800000); // -1.0f
        st->tmpbuf[ncfft - k].i = __lsx_vfmul_s(st->tmpbuf[ncfft - k].i, neg_one);
#else
        st->tmpbuf[ncfft - k].i *= _mm_set1_ps(-1.0);
#endif
#else
        st->tmpbuf[ncfft - k].i *= -1;
#endif
    }
    kiss_fft (st->substate, st->tmpbuf, (kiss_fft_cpx *) timedata);
}

/* ------------------------------------------------------------------ */
/* Config reuse pool, in-place entries and batch processing.          */
/* ------------------------------------------------------------------ */

struct kiss_fftr_pool_entry {
    int nfft;
    int inverse;
    int in_use;
    struct kiss_fftr_pool_entry *next;
    kiss_fftr_cfg cfg; /* embedded right after this header, one block */
};

struct kiss_fftr_pool {
    struct kiss_fftr_pool_entry *head;
};

kiss_fftr_pool_t kiss_fftr_pool_create(void)
{
    kiss_fftr_pool_t pool =
        (kiss_fftr_pool_t) KISS_FFT_MALLOC(sizeof(struct kiss_fftr_pool));
    if (pool == NULL) {
        KISS_FFT_ERROR("kiss_fftr_pool_create: allocation failed.");
        return NULL;
    }
    pool->head = NULL;
    return pool;
}

void kiss_fftr_pool_destroy(kiss_fftr_pool_t pool)
{
    struct kiss_fftr_pool_entry *e;
    if (pool == NULL)
        return;
    e = pool->head;
    while (e != NULL) {
        struct kiss_fftr_pool_entry *next = e->next;
        /* entry header and embedded cfg live in a single allocation */
        KISS_FFT_FREE(e);
        e = next;
    }
    KISS_FFT_FREE(pool);
}

kiss_fftr_cfg kiss_fftr_pool_acquire(kiss_fftr_pool_t pool,
                                     int nfft,
                                     int inverse_fft)
{
    struct kiss_fftr_pool_entry *e;
    size_t cfgsize = 0, need;
    char *block;

    if (pool == NULL || nfft <= 0 || (nfft & 1))
        return NULL;

    for (e = pool->head; e != NULL; e = e->next) {
        if (!e->in_use && e->nfft == nfft && e->inverse == inverse_fft) {
            e->in_use = 1;
            return e->cfg;
        }
    }

    /* find out how large the real-FFT config is, without allocating it */
    if (kiss_fftr_alloc(nfft, inverse_fft, NULL, &cfgsize) == NULL &&
        cfgsize == 0)
        return NULL; /* invalid nfft */

    /* one allocation holds the entry header and the whole config block */
    need = sizeof(struct kiss_fftr_pool_entry) + cfgsize;
    block = (char *) KISS_FFT_MALLOC(need);
    if (block == NULL) {
        KISS_FFT_ERROR("kiss_fftr_pool_acquire: allocation failed.");
        return NULL;
    }
    e = (struct kiss_fftr_pool_entry *) block;
    e->cfg = (kiss_fftr_cfg) (block + sizeof(struct kiss_fftr_pool_entry));
    if (kiss_fftr_alloc(nfft, inverse_fft, e->cfg, &cfgsize) == NULL) {
        KISS_FFT_FREE(e);
        return NULL;
    }
    e->nfft = nfft;
    e->inverse = inverse_fft;
    e->in_use = 1;
    e->next = pool->head;
    pool->head = e;
    return e->cfg;
}

void kiss_fftr_pool_release(kiss_fftr_pool_t pool, kiss_fftr_cfg cfg)
{
    struct kiss_fftr_pool_entry *e;
    if (pool == NULL || cfg == NULL)
        return;
    for (e = pool->head; e != NULL; e = e->next) {
        if (e->cfg == cfg) {
            e->in_use = 0;
            return;
        }
    }
    KISS_FFT_ERROR("kiss_fftr_pool_release: cfg does not belong to pool.");
}

void kiss_fftr_inplace(kiss_fftr_cfg st, kiss_fft_scalar *buf)
{
    /* The forward transform only reads timedata through the nfft/2-point
     * complex sub-FFT and the post-processing loop, both of which finish
     * before freqdata (which starts at the same address) is written. The
     * state's own tmpbuf holds every intermediate, so reusing the caller
     * buffer for the output needs no extra allocation. */
    kiss_fftr(st, buf, (kiss_fft_cpx *) buf);
}

void kiss_fftr_inplacei(kiss_fftr_cfg st, kiss_fft_scalar *buf)
{
    /* The inverse first builds tmpbuf from freqdata and only then runs
     * the sub-FFT into timedata, so sharing one buffer is safe. */
    kiss_fftri(st, (const kiss_fft_cpx *) buf, buf);
}

/* Run one transform per segment, grouping segments of equal nfft so the
 * config for a length is acquired once and reused for the whole group.
 * No scratch storage is allocated: each segment only checks whether an
 * earlier segment already represents its length. */
static int kiss_fftr_batch_run(kiss_fftr_pool_t pool,
                               const kiss_fftr_batch_seg *segs,
                               int nseg,
                               int inverse_fft)
{
    int i, j, done = 0;
    if (pool == NULL || segs == NULL || nseg <= 0)
        return 0;

    /* fast path: the common "N equal-length segments" case needs no
     * grouping scan at all -- one config for the whole batch */
    {
        int common = segs[0].nfft;
        int uniform = (common > 0 && !(common & 1) &&
                       segs[0].timedata != NULL && segs[0].freqdata != NULL);
        for (i = 1; i < nseg && uniform; ++i)
            uniform = (segs[i].nfft == common &&
                       segs[i].timedata != NULL && segs[i].freqdata != NULL);
        if (uniform) {
            kiss_fftr_cfg cfg = kiss_fftr_pool_acquire(pool, common,
                                                       inverse_fft);
            if (cfg == NULL)
                return 0;
            for (i = 0; i < nseg; ++i) {
                if (inverse_fft)
                    kiss_fftri(cfg, segs[i].freqdata, segs[i].timedata);
                else
                    kiss_fftr(cfg, segs[i].timedata, segs[i].freqdata);
            }
            kiss_fftr_pool_release(pool, cfg);
            return nseg;
        }
    }

    /* general path: mixed lengths, grouped without extra allocation.
     * A valid earlier group leader always has non-NULL buffers, so a
     * skipped (invalid) index cannot be the leader of a later length. */
    for (i = 0; i < nseg; ++i) {
        kiss_fftr_cfg cfg;
        if (segs[i].nfft <= 0 || (segs[i].nfft & 1) ||
            segs[i].timedata == NULL || segs[i].freqdata == NULL)
            continue;
        for (j = 0; j < i; ++j) {
            if (segs[j].nfft == segs[i].nfft)
                break;
        }
        if (j < i)
            continue;

        cfg = kiss_fftr_pool_acquire(pool, segs[i].nfft, inverse_fft);
        if (cfg == NULL)
            continue;
        for (j = i; j < nseg; ++j) {
            if (segs[j].nfft != segs[i].nfft)
                continue;
            if (inverse_fft)
                kiss_fftri(cfg, segs[j].freqdata, segs[j].timedata);
            else
                kiss_fftr(cfg, segs[j].timedata, segs[j].freqdata);
            ++done;
        }
        kiss_fftr_pool_release(pool, cfg);
    }
    return done;
}

int kiss_fftr_batch(kiss_fftr_pool_t pool,
                    const kiss_fftr_batch_seg *segs,
                    int nseg)
{
    return kiss_fftr_batch_run(pool, segs, nseg, 0);
}

int kiss_fftr_batchi(kiss_fftr_pool_t pool,
                     const kiss_fftr_batch_seg *segs,
                     int nseg)
{
    return kiss_fftr_batch_run(pool, segs, nseg, 1);
}
