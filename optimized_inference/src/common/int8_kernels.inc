/* Shared AVX2/VNNI kernels. Instantiated in two translation units so the
 * AVX2 path never contains instructions unavailable on an older CPU.
 * Quantization is unchanged: u8 activations <=254 and signed weights <=63.
 * Pair products fit i16, and K<=512 bounds the complete sum within i32. */
#ifdef DPDF_INT8_VNNI
#define DPDF_DOT4(sum,a,w) _mm256_dpbusd_avx_epi32(sum,a,w)
#include "int8_vnni_aliases.h"
#else
#define DPDF_DOT4(sum,a,w) _mm256_add_epi32(sum,_mm256_madd_epi16( \
    _mm256_maddubs_epi16(a,w),_mm256_set1_epi16(1)))
#endif

/* Row range. The scalar reference starts at zero and ignores unordered
 * comparisons; a zero second operand preserves that behavior, so neither
 * bound is NaN. */
static inline __attribute__((always_inline)) void quantize_range(
        const float *x,int k,float *low,float *high) {
    __m256 lo=_mm256_setzero_ps(),hi=lo;
    for (int j=0;j<k;j+=8) { __m256 v=_mm256_loadu_ps(x+j); lo=_mm256_min_ps(lo,v); hi=_mm256_max_ps(hi,v); }
    lo=_mm256_min_ps(lo,_mm256_setzero_ps());
    hi=_mm256_max_ps(hi,_mm256_setzero_ps());
    __m128 low4=_mm_min_ps(_mm256_castps256_ps128(lo),_mm256_extractf128_ps(lo,1));
    __m128 high4=_mm_max_ps(_mm256_castps256_ps128(hi),_mm256_extractf128_ps(hi,1));
    low4=_mm_min_ps(low4,_mm_movehl_ps(low4,low4));
    high4=_mm_max_ps(high4,_mm_movehl_ps(high4,high4));
    *low=_mm_cvtss_f32(_mm_min_ss(low4,_mm_shuffle_ps(low4,low4,0x55)));
    *high=_mm_cvtss_f32(_mm_max_ss(high4,_mm_shuffle_ps(high4,high4,0x55)));
}

/* Row parameters: 0 for a constant row, whose activations are all zero. */
static inline __attribute__((always_inline)) int quantize_params(
        const float *x,int k,float *scale,int *zp,float *inverse) {
    float low,high;
    quantize_range(x,k,&low,&high);
    if (low==high) { *scale=1; *zp=0; return 0; }
    *scale=(high-low)/254.0f;
    *zp=(int)nearbyintf(-low / *scale);
    if (*zp<0) *zp=0;
    if (*zp>254) *zp=254;
    *inverse=1.0f / *scale;
    return 1;
}

static inline __attribute__((always_inline)) void quantize_apply(
        const float *x,int8_t *out,int k,float inverse,int zp) {
    __m256 inv=_mm256_set1_ps(inverse),z=_mm256_set1_ps((float)zp);
    int j=0;
    for (;j+31<k;j+=32) {
        __m256i q[4];
        for (int t=0;t<4;++t) {
            __m256 v=_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x+j+t*8),inv),z);
            q[t]=_mm256_cvtps_epi32(v);
        }
        __m256i p=_mm256_packus_epi16(_mm256_packs_epi32(q[0],q[1]),_mm256_packs_epi32(q[2],q[3]));
        /* The two packs saturate to [0,255]; one byte min gives [0,254]. */
        p=_mm256_min_epu8(p,_mm256_set1_epi8((char)254));
        /* AVX2 packs within 128-bit lanes: reorder four-byte groups. */
        p=_mm256_permutevar8x32_epi32(p,_mm256_setr_epi32(0,4,1,5,2,6,3,7));
        _mm256_storeu_si256((__m256i *)(out+j),p);
    }
    for (;j<k;j+=8) {
        __m256 v=_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x+j),inv),z);
        __m256i q=_mm256_cvtps_epi32(v);
        __m128i p=_mm_packs_epi32(_mm256_castsi256_si128(q),_mm256_extracti128_si256(q,1));
        p=_mm_packus_epi16(p,p);
        p=_mm_min_epu8(p,_mm_set1_epi8((char)254));
        _mm_storel_epi64((__m128i *)(out+j),p);
    }
}

static void quantize(const float *x,int8_t *out,int k,float *scale,int *zp) {
    float inverse;
    if (quantize_params(x,k,scale,zp,&inverse)) quantize_apply(x,out,k,inverse,*zp);
    else memset(out,0,k);
}

static inline __attribute__((always_inline,unused)) void quantize64(
        const float *x,int8_t *out,float *scale,int *zp) {
    float inverse;
    if (quantize_params(x,64,scale,zp,&inverse)) quantize_apply(x,out,64,inverse,*zp);
    else memset(out,0,64);
}

/* quantize_params for eight rows per vector, with the same IEEE operations.
 * Constant rows divide a placeholder (0,1) range, raising no new exceptions,
 * and keep scale 1 and zero point 0; *constant marks them. */
static inline __attribute__((always_inline)) void quantize_params8(
        __m256 lo,__m256 hi,__m256 *scale_out,__m256i *zp_out,__m256 *inverse,__m256 *constant) {
    const __m256 zero=_mm256_setzero_ps(),one=_mm256_set1_ps(1.0f);
    __m256 flat=_mm256_cmp_ps(lo,hi,_CMP_EQ_OQ);
    lo=_mm256_blendv_ps(lo,zero,flat); hi=_mm256_blendv_ps(hi,one,flat);
    __m256 scale=_mm256_div_ps(_mm256_sub_ps(hi,lo),_mm256_set1_ps(254.0f));
    __m256 point=_mm256_round_ps(_mm256_div_ps(_mm256_xor_ps(lo,_mm256_set1_ps(-0.0f)),scale),
                                 _MM_FROUND_CUR_DIRECTION|_MM_FROUND_NO_EXC);
    __m256i z=_mm256_cvttps_epi32(point);
    z=_mm256_min_epi32(_mm256_max_epi32(z,_mm256_setzero_si256()),_mm256_set1_epi32(254));
    *inverse=_mm256_div_ps(one,scale);
    *scale_out=_mm256_blendv_ps(scale,one,flat);
    *zp_out=_mm256_andnot_si256(_mm256_castps_si256(flat),z);
    *constant=flat;
}

/* Batches find every row's range first, then derive parameters eight rows at
 * a time. The serial range/division chain otherwise stalls conversions. */
static inline __attribute__((always_inline)) void quantize_rows_k(
        const float *x,int8_t *out,int k,int m,float *scales,int *zp) {
    float low[48],high[48],inverse[48]; int live[48];
    for (int r=0;r<m;++r) quantize_range(x+r*k,k,low+r,high+r);
    for (int r=m;r<(m+7)/8*8;++r) { low[r]=0; high[r]=1; }
    for (int r=0;r<m;r+=8) {
        __m256 scale,inv,constant; __m256i z;
        quantize_params8(_mm256_loadu_ps(low+r),_mm256_loadu_ps(high+r),&scale,&z,&inv,&constant);
        float s8[8],i8[8]; int z8[8],l8[8];
        _mm256_storeu_ps(s8,scale);
        _mm256_storeu_si256((__m256i *)z8,z);
        _mm256_storeu_ps(i8,inv);
        _mm256_storeu_si256((__m256i *)l8,_mm256_andnot_si256(_mm256_castps_si256(constant),_mm256_set1_epi32(1)));
        for (int t=0;t<8 && r+t<m;++t) { scales[r+t]=s8[t]; zp[r+t]=z8[t]; inverse[r+t]=i8[t]; live[r+t]=l8[t]; }
    }
    for (int r=0;r<m;++r) {
        if (live[r]) quantize_apply(x+r*k,out+r*k,k,inverse[r],zp[r]);
        else memset(out+r*k,0,k);
    }
}
static void quantize_rows(const float *x,int8_t *out,int k,int m,float *scales,int *zp) {
    if (k==64) quantize_rows_k(x,out,64,m,scales,zp);
    else quantize_rows_k(x,out,k,m,scales,zp);
}

#ifndef DPDF_DISABLE_QAFFINE_ROW
#ifdef DPDF_QASM64
void dpdf_qdot64_fixed_avx2(const int8_t *,const int8_t *,int32_t *);
#if DPDF_MODEL_SIZE == 2
#ifdef DPDF_QFUSED256
void dpdf_qaffine256_fused_avx2(const int8_t *,const int8_t *,const int32_t *,
        const float *,const float *,float *,float,int,int);
#endif
#else
#endif
#ifdef DPDF_QFUSED64
void dpdf_qaffine64_fused_avx2(const int8_t *,const int8_t *,const int32_t *,
        const float *,const float *,float *,float,int);
#endif
#endif
/* Keep integer reduction separate from quantization/dequantization: inlining
 * their live temporaries caused GCC 12 to spill two accumulators every K step.
 * Short weight lifetimes let all eight accumulators stay in registers. The
 * noinline boundary is intentional; both instruction variants retain the same reduction order.
 * Activations are [0,254], weights are [-63,63]: pair sums fit i16 (32004).
 * K<=512 also keeps the full dot product safely within i32. */
static __attribute__((noinline)) void qdot64(const int8_t *activation,
        const int8_t *packed,int k,int32_t *out) {
    __m256i sum0=_mm256_setzero_si256(),sum1=sum0,sum2=sum0,sum3=sum0;
    __m256i sum4=sum0,sum5=sum0,sum6=sum0,sum7=sum0;
    for (int j=0;j<k;j+=4) {
        int32_t bytes; memcpy(&bytes,activation+j,4);
        __m256i a=_mm256_set1_epi32(bytes),absolute=a;
        const int8_t *base=packed+j*8;
        __m256i w0=_mm256_loadu_si256((const __m256i *)(base));
        sum0=DPDF_DOT4(sum0,absolute,w0);
        __m256i w1=_mm256_loadu_si256((const __m256i *)(base+k*8));
        sum1=DPDF_DOT4(sum1,absolute,w1);
        __m256i w2=_mm256_loadu_si256((const __m256i *)(base+2*k*8));
        sum2=DPDF_DOT4(sum2,absolute,w2);
        __m256i w3=_mm256_loadu_si256((const __m256i *)(base+3*k*8));
        sum3=DPDF_DOT4(sum3,absolute,w3);
        __m256i w4=_mm256_loadu_si256((const __m256i *)(base+4*k*8));
        sum4=DPDF_DOT4(sum4,absolute,w4);
        __m256i w5=_mm256_loadu_si256((const __m256i *)(base+5*k*8));
        sum5=DPDF_DOT4(sum5,absolute,w5);
        __m256i w6=_mm256_loadu_si256((const __m256i *)(base+6*k*8));
        sum6=DPDF_DOT4(sum6,absolute,w6);
        __m256i w7=_mm256_loadu_si256((const __m256i *)(base+7*k*8));
        sum7=DPDF_DOT4(sum7,absolute,w7);
    }
    _mm256_storeu_si256((__m256i *)(out+0),sum0);
    _mm256_storeu_si256((__m256i *)(out+8),sum1);
    _mm256_storeu_si256((__m256i *)(out+16),sum2);
    _mm256_storeu_si256((__m256i *)(out+24),sum3);
    _mm256_storeu_si256((__m256i *)(out+32),sum4);
    _mm256_storeu_si256((__m256i *)(out+40),sum5);
    _mm256_storeu_si256((__m256i *)(out+48),sum6);
    _mm256_storeu_si256((__m256i *)(out+56),sum7);
}

static void qaffine_row(const dpdf_qmatrix *q,const float *x,const float *bias,float *y) {
    int8_t activation[512]; float activation_scale; int zp;
    const int k=q->k,n=q->n;
#if DPDF_MODEL_SIZE == 2
    quantize(x,activation,k,&activation_scale,&zp);
#ifdef DPDF_QFUSED256
    if (k==256 && n==768) {
        dpdf_qaffine256_fused_avx2(activation,q->packed,q->sum,q->scale,bias,y,
                                 activation_scale,zp,12);
        return;
    }
#endif
#else
    if (k==64) quantize64(x,activation,&activation_scale,&zp);
    else quantize(x,activation,k,&activation_scale,&zp);
#endif
#ifdef DPDF_QFUSED64
    if (k==64 && n==192) {
        dpdf_qaffine64_fused_avx2(activation,q->packed,q->sum,q->scale,bias,y,
                                activation_scale,zp);
        return;
    }
#endif
    int c=0;
    for (;c+63<n;c+=64) {
        int32_t integer_sums[64];
#ifdef DPDF_QASM64
        if (k==64) dpdf_qdot64_fixed_avx2(activation,q->packed+c*k,integer_sums);
        else
#endif
            qdot64(activation,q->packed+c*k,k,integer_sums);
        __m256i sums[8];
        for (int t=0;t<8;++t) sums[t]=_mm256_loadu_si256((const __m256i *)(integer_sums+t*8));
        for (int t=0;t<8;++t) {
            int column=c+t*8;
            __m256i correction=_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i *)(q->sum+column)),_mm256_set1_epi32(-zp));
            __m256 sum=_mm256_cvtepi32_ps(_mm256_add_epi32(sums[t],correction));
            __m256 scale=_mm256_mul_ps(_mm256_set1_ps(activation_scale),_mm256_loadu_ps(q->scale+column));
            _mm256_storeu_ps(y+column,_mm256_fmadd_ps(sum,scale,_mm256_loadu_ps(bias+column)));
        }
    }
    for (;c<n;c+=8) {
        __m256i sum=_mm256_setzero_si256();
        for (int j=0;j<k;j+=4) {
            int32_t bytes; memcpy(&bytes,activation+j,4);
            __m256i a=_mm256_set1_epi32(bytes);
            __m256i w=_mm256_loadu_si256((const __m256i *)(q->packed+c*k+j*8));
            sum=DPDF_DOT4(sum,a,w);
        }
        sum=_mm256_add_epi32(sum,_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i *)(q->sum+c)),_mm256_set1_epi32(-zp)));
        __m256 scale=_mm256_mul_ps(_mm256_set1_ps(activation_scale),_mm256_loadu_ps(q->scale+c));
        _mm256_storeu_ps(y+c,_mm256_fmadd_ps(_mm256_cvtepi32_ps(sum),scale,_mm256_loadu_ps(bias+c)));
    }
}
#endif

/* Four rows and two independent 8-column tiles. Keeping the integer loop in
 * its own function leaves room for eight accumulators and shared weights;
 * scale/correction temporaries are only live after the reduction returns. */
static __attribute__((noinline)) void qdot4pair_generic(const int8_t *activation,
        const int8_t *packed0,const int8_t *packed1,int k,__m256i *out) {
    __m256i a0=_mm256_setzero_si256(),a1=a0,a2=a0,a3=a0;
    __m256i b0=a0,b1=a0,b2=a0,b3=a0;
    for (int j=0;j<k;j+=4) {
        __m256i w0=_mm256_loadu_si256((const __m256i *)(packed0+j*8));
        __m256i w1=_mm256_loadu_si256((const __m256i *)(packed1+j*8));
#define DPDF_DOT_ROW(row,sa,sb) do { \
        int32_t bytes; memcpy(&bytes,activation+(row)*k+j,4); \
        __m256i v=_mm256_set1_epi32(bytes),absolute=v; \
        sa=DPDF_DOT4(sa,absolute,w0); \
        sb=DPDF_DOT4(sb,absolute,w1); \
    } while (0)
        DPDF_DOT_ROW(0,a0,b0); DPDF_DOT_ROW(1,a1,b1);
        DPDF_DOT_ROW(2,a2,b2); DPDF_DOT_ROW(3,a3,b3);
#undef DPDF_DOT_ROW
    }
    out[0]=a0;out[1]=a1;out[2]=a2;out[3]=a3;
    out[4]=b0;out[5]=b1;out[6]=b2;out[7]=b3;
}

#ifdef DPDF_QPAIR_FIXED
void dpdf_qdot4pair64_avx2(const int8_t *,const int8_t *,const int8_t *,__m256i *);
#endif
static __attribute__((noinline)) void qdot4pair(const int8_t *activation,
        const int8_t *packed0,const int8_t *packed1,int k,__m256i *out) {
#ifdef DPDF_QPAIR_FIXED
    if (k==64) { dpdf_qdot4pair64_avx2(activation,packed0,packed1,out); return; }
#endif
    qdot4pair_generic(activation,packed0,packed1,k,out);
}

/* Model 8 retains the faster original tiled reduction for AVX2-only CPUs.
 * The shared fused four-row kernel is selected there only for AVX-VNNI.
 * Model 2 keeps its validated fused path for both ISA variants. */
#if defined(DPDF_QPAIR_FUSED64) && (DPDF_MODEL_SIZE == 2 || defined(DPDF_INT8_VNNI))
#include "pair_fused_args.h"
static inline __attribute__((always_inline)) void qpair_fused4(
        const dpdf_qmatrix *q0,const dpdf_qmatrix *q1,const int8_t *activation,
        const float *scales,const int *zp,const float *bias0,const float *bias1,
        float *y0,float *y1,int c0,int c1) {
    const dpdf_pair_fused_args args={
        activation,q0->packed+c0*64,q1->packed+c1*64,q0->sum+c0,q1->sum+c1,
        q0->scale+c0,q1->scale+c1,bias0+c0,bias1+c1,y0+c0,y1+c1,
        (size_t)q0->n*sizeof(float),
        {scales[0],scales[1],scales[2],scales[3]},
        {zp[0],zp[1],zp[2],zp[3]}
    };
    dpdf_qaffine4pair64_fused_avx2(&args);
}
#endif
static void qaffine_batch_tiled(const dpdf_qmatrix *q,const float *x,const float *bias,float *y,int m) {
    int8_t activation[48*512]; float scales[48]; int zp[48];
    const int k=q->k,n=q->n;
    quantize_rows(x,activation,k,m,scales,zp);
#if defined(DPDF_QPAIR_FUSED64) && (DPDF_MODEL_SIZE == 2 || defined(DPDF_INT8_VNNI))
    if (k==64) {
        for (int r=0;r<m;r+=4) for (int c=0;c<n;c+=16)
            qpair_fused4(q,q,activation+r*64,scales+r,zp+r,bias,bias,
                         y+r*n,y+r*n,c,c+8);
        return;
    }
#endif
    for (int r=0;r<m;r+=4) for (int c=0;c<n;c+=16) {
        __m256i sums[8];
        qdot4pair(activation+r*k,q->packed+c*k,q->packed+(c+8)*k,k,sums);
        for (int t=0;t<4;++t) for (int tile=0;tile<2;++tile) {
            int col=c+tile*8;
            __m256i correction=_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i *)(q->sum+col)),_mm256_set1_epi32(-zp[r+t]));
            __m256 scale=_mm256_mul_ps(_mm256_set1_ps(scales[r+t]),_mm256_loadu_ps(q->scale+col));
            __m256 result=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sums[t+tile*4],correction)),scale,_mm256_loadu_ps(bias+col));
            _mm256_storeu_ps(y+(r+t)*n+col,result);
        }
    }
}

static void qaffine_batch(const dpdf_qmatrix *q,const float *x,const float *bias,float *y,int m) {
    if (m%4==0 && q->n%16==0) { qaffine_batch_tiled(q,x,bias,y,m); return; }
    /* Callers split larger batches into at most 48 rows. */
    int8_t activation[48*512]; float scales[48]; int zp[48];
    const int k=q->k,n=q->n;
    quantize_rows(x,activation,k,m,scales,zp);
    int r=0;
    for (;r+3<m;r+=4) for (int c=0;c<n;c+=8) {
        __m256i s0=_mm256_setzero_si256(),s1=s0,s2=s0,s3=s0;
        for (int j=0;j<k;j+=4) {
            int32_t v0,v1,v2,v3;
            memcpy(&v0,activation+r*k+j,4); memcpy(&v1,activation+(r+1)*k+j,4);
            memcpy(&v2,activation+(r+2)*k+j,4); memcpy(&v3,activation+(r+3)*k+j,4);
            __m256i a0=_mm256_set1_epi32(v0),a1=_mm256_set1_epi32(v1);
            __m256i a2=_mm256_set1_epi32(v2),a3=_mm256_set1_epi32(v3);
            __m256i w=_mm256_loadu_si256((const __m256i *)(q->packed+c*k+j*8));
            s0=DPDF_DOT4(s0,a0,w);
            s1=DPDF_DOT4(s1,a1,w);
            s2=DPDF_DOT4(s2,a2,w);
            s3=DPDF_DOT4(s3,a3,w);
        }
        __m256i sums[4]={s0,s1,s2,s3};
        for (int t=0;t<4;++t) {
            __m256i correction=_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i *)(q->sum+c)),_mm256_set1_epi32(-zp[r+t]));
            __m256 scale=_mm256_mul_ps(_mm256_set1_ps(scales[r+t]),_mm256_loadu_ps(q->scale+c));
            __m256 result=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sums[t],correction)),scale,_mm256_loadu_ps(bias+c));
            _mm256_storeu_ps(y+(r+t)*n+c,result);
        }
    }
    for (;r<m;++r) {
      int c=0;
      /* Recurrent GRU projections arrive here one row at a time. Walk K once
       * for eight output vectors so each activation load/broadcast is reused.
       * Each accumulator retains the original increasing-K sum order. */
      for (;c+63<n;c+=64) {
        __m256i sum0=_mm256_setzero_si256(),sum1=sum0,sum2=sum0,sum3=sum0;
        __m256i sum4=sum0,sum5=sum0,sum6=sum0,sum7=sum0;
        for (int j=0;j<k;j+=4) {
            int32_t bytes; memcpy(&bytes,activation+r*k+j,4);
            __m256i a=_mm256_set1_epi32(bytes);
            __m256i absolute=a;
            const int8_t *base=q->packed+(c/8)*k*8+j*8;
            __m256i w0=_mm256_loadu_si256((const __m256i *)(base));
            __m256i w1=_mm256_loadu_si256((const __m256i *)(base+k*8));
            __m256i w2=_mm256_loadu_si256((const __m256i *)(base+2*k*8));
            __m256i w3=_mm256_loadu_si256((const __m256i *)(base+3*k*8));
            __m256i w4=_mm256_loadu_si256((const __m256i *)(base+4*k*8));
            __m256i w5=_mm256_loadu_si256((const __m256i *)(base+5*k*8));
            __m256i w6=_mm256_loadu_si256((const __m256i *)(base+6*k*8));
            __m256i w7=_mm256_loadu_si256((const __m256i *)(base+7*k*8));
            /* 2*254*63=32004: AVX2 never saturates, so VNNI is exact. */
            sum0=DPDF_DOT4(sum0,absolute,w0);
            sum1=DPDF_DOT4(sum1,absolute,w1);
            sum2=DPDF_DOT4(sum2,absolute,w2);
            sum3=DPDF_DOT4(sum3,absolute,w3);
            sum4=DPDF_DOT4(sum4,absolute,w4);
            sum5=DPDF_DOT4(sum5,absolute,w5);
            sum6=DPDF_DOT4(sum6,absolute,w6);
            sum7=DPDF_DOT4(sum7,absolute,w7);
        }
        __m256i sums[8]={sum0,sum1,sum2,sum3,sum4,sum5,sum6,sum7};
        for (int t=0;t<8;++t) {
            int column=c+t*8;
            __m256i correction=_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i *)(q->sum+column)),_mm256_set1_epi32(-zp[r]));
            __m256 sum=_mm256_cvtepi32_ps(_mm256_add_epi32(sums[t],correction));
            __m256 scale=_mm256_mul_ps(_mm256_set1_ps(scales[r]),_mm256_loadu_ps(q->scale+column));
            _mm256_storeu_ps(y+r*n+column,_mm256_fmadd_ps(sum,scale,_mm256_loadu_ps(bias+column)));
        }
      }
      for (;c<n;c+=8) {
        __m256i sum=_mm256_setzero_si256();
        for (int j=0;j<k;j+=4) {
            int32_t bytes; memcpy(&bytes,activation+r*k+j,4);
            __m256i a=_mm256_set1_epi32(bytes);
            __m256i w=_mm256_loadu_si256((const __m256i *)(q->packed+c*k+j*8));
            sum=DPDF_DOT4(sum,a,w);
        }
        __m256i correction=_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i *)(q->sum+c)),_mm256_set1_epi32(-zp[r]));
        sum=_mm256_add_epi32(sum,correction);
        __m256 scale=_mm256_mul_ps(_mm256_set1_ps(scales[r]),_mm256_loadu_ps(q->scale+c));
        __m256 result=_mm256_fmadd_ps(_mm256_cvtepi32_ps(sum),scale,_mm256_loadu_ps(bias+c));
        _mm256_storeu_ps(y+r*n+c,result);
      }
    }
}

void DPDF_QAFFINE_IMPL(const dpdf_qmatrix *q,const float *x,const float *bias,float *y,int m) {
#ifndef DPDF_DISABLE_QAFFINE_ROW
    if (m==1) qaffine_row(q,x,bias,y);
    else qaffine_batch(q,x,bias,y,m);
#else
    qaffine_batch(q,x,bias,y,m);
#endif
}

/* The bidirectional intra-frequency GRU applies two matrices to exactly the
 * same rows. Quantize once and walk both packed matrices together, retaining
 * the original increasing-K accumulation order for every output. */
void DPDF_QPAIR_IMPL(const dpdf_qmatrix *q0,const dpdf_qmatrix *q1,const float *x,
                       const float *bias0,const float *bias1,float *y0,float *y1,int m) {
    if (q0->k!=q1->k || q0->n!=q1->n || m%4) {
        DPDF_QAFFINE_IMPL(q0,x,bias0,y0,m); DPDF_QAFFINE_IMPL(q1,x,bias1,y1,m); return;
    }
    int8_t activation[48*512]; float scales[48]; int zp[48];
    const int k=q0->k,n=q0->n;
    quantize_rows(x,activation,k,m,scales,zp);
#if defined(DPDF_QPAIR_FUSED64) && (DPDF_MODEL_SIZE == 2 || defined(DPDF_INT8_VNNI))
    if (k==64) {
        for (int r=0;r<m;r+=4) for (int c=0;c<n;c+=8)
            qpair_fused4(q0,q1,activation+r*64,scales+r,zp+r,bias0,bias1,
                         y0+r*n,y1+r*n,c,c);
        return;
    }
#endif
    for (int r=0;r<m;r+=4) for (int c=0;c<n;c+=8) {
        __m256i sums[8];
        qdot4pair(activation+r*k,q0->packed+c*k,q1->packed+c*k,k,sums);
        const __m256i weight_sum0=_mm256_loadu_si256((const __m256i *)(q0->sum+c));
        const __m256i weight_sum1=_mm256_loadu_si256((const __m256i *)(q1->sum+c));
        const __m256 weight_scale0=_mm256_loadu_ps(q0->scale+c);
        const __m256 weight_scale1=_mm256_loadu_ps(q1->scale+c);
        for (int t=0;t<4;++t) {
            __m256i zero=_mm256_set1_epi32(-zp[r+t]);
            __m256 scale=_mm256_set1_ps(scales[r+t]);
            __m256i corrected=_mm256_add_epi32(sums[t],_mm256_mullo_epi32(weight_sum0,zero));
            _mm256_storeu_ps(y0+(r+t)*n+c,_mm256_fmadd_ps(_mm256_cvtepi32_ps(corrected),
                             _mm256_mul_ps(scale,weight_scale0),_mm256_loadu_ps(bias0+c)));
            corrected=_mm256_add_epi32(sums[t+4],_mm256_mullo_epi32(weight_sum1,zero));
            _mm256_storeu_ps(y1+(r+t)*n+c,_mm256_fmadd_ps(_mm256_cvtepi32_ps(corrected),
                             _mm256_mul_ps(scale,weight_scale1),_mm256_loadu_ps(bias1+c)));
        }
    }
}

/* Bidirectional intra-frequency GRU recurrence for K=64, N=192 matrices:
 * direction 0 walks rows 0..f-1 with q0, direction 1 walks f-1..0 with q1.
 * Each step quantizes h, projects it, applies the gates and stores h in
 * bi[r*128+direction*64]. The directions are independent, so the steps are
 * software-pipelined: one direction's latency-bound gates and quantization
 * sit just before the other direction's throughput-bound projection. Every
 * direction keeps its exact operation sequence. */
void DPDF_QRECUR_IMPL(const dpdf_qmatrix *q0,const dpdf_qmatrix *q1,
        const float *intra0,const float *intra1,const float *bias0,const float *bias1,
        int f,float *bi) {
    float h0[64],h1[64],step0[192],step1[192];
    _Alignas(32) int8_t act0[64],act1[64];
    float scale0,scale1; int zp0,zp1;
    memset(h0,0,sizeof(h0)); memset(h1,0,sizeof(h1));
    for (int c=0;c<192;++c) { step0[c]=0.0f+bias0[c]; step1[c]=0.0f+bias1[c]; }
    dpdf_gates_avx2(intra0,step0,h0,h0,1);
    memcpy(bi,h0,sizeof(h0));
    dpdf_gates_avx2(intra1+(f-1)*192,step1,h1,h1,1);
    memcpy(bi+(f-1)*128+64,h1,sizeof(h1));
    if (f>1) quantize64(h0,act0,&scale0,&zp0);
    for (int t=1;t<f;++t) {
        const int r1=f-1-t;
        quantize64(h1,act1,&scale1,&zp1);
        dpdf_qaffine64_fused_avx2(act0,q0->packed,q0->sum,q0->scale,bias0,step0,scale0,zp0);
        dpdf_gates_avx2(intra0+t*192,step0,h0,h0,1);
        memcpy(bi+t*128,h0,sizeof(h0));
        if (t+1<f) quantize64(h0,act0,&scale0,&zp0);
        dpdf_qaffine64_fused_avx2(act1,q1->packed,q1->sum,q1->scale,bias1,step1,scale1,zp1);
        dpdf_gates_avx2(intra1+r1*192,step1,h1,h1,1);
        memcpy(bi+r1*128+64,h1,sizeof(h1));
    }
}

/* Channel-major pointwise projection for K=N=64: each of count (1-3) pieces
 * is [64][width]; y is [64][count*width] and position t*count+j is column t
 * of piece j. Quantization, integer sums and the FP32 epilogue match the
 * row-major path for every position, so the graph's surrounding transposes
 * are unnecessary. With relu, outputs take x>0?x:0 (max(x,0) selects the
 * same bits). All positions are quantized before y is written, so y may
 * overlap the pieces. */
#ifdef DPDF_INT8_VNNI
#define DPDF_CM_VECTORS 3 /* 4x3 accumulators hide VNNI latency. */
#else
#define DPDF_CM_VECTORS 2 /* AVX2 needs a ones vector and temporary. */
#endif
enum { DPDF_CM_MAX=480 };

/* out lane l of vector i holds v[(8i+l)%count] lane (8i+l)/count. */
static inline __attribute__((always_inline)) void interleave32(const __m256i *v,int count,__m256i *out) {
    if (count==1) { out[0]=v[0]; return; }
    if (count==2) {
        __m256i lo=_mm256_unpacklo_epi32(v[0],v[1]),hi=_mm256_unpackhi_epi32(v[0],v[1]);
        out[0]=_mm256_permute2x128_si256(lo,hi,0x20);
        out[1]=_mm256_permute2x128_si256(lo,hi,0x31);
        return;
    }
    const __m256i i0=_mm256_setr_epi32(0,0,0,1,1,1,2,2);
    const __m256i i1=_mm256_setr_epi32(2,3,3,3,4,4,4,5);
    const __m256i i2=_mm256_setr_epi32(5,5,6,6,6,7,7,7);
    out[0]=_mm256_blend_epi32(_mm256_blend_epi32(_mm256_permutevar8x32_epi32(v[0],i0),
           _mm256_permutevar8x32_epi32(v[1],i0),0x92),_mm256_permutevar8x32_epi32(v[2],i0),0x24);
    out[1]=_mm256_blend_epi32(_mm256_blend_epi32(_mm256_permutevar8x32_epi32(v[0],i1),
           _mm256_permutevar8x32_epi32(v[1],i1),0x24),_mm256_permutevar8x32_epi32(v[2],i1),0x49);
    out[2]=_mm256_blend_epi32(_mm256_blend_epi32(_mm256_permutevar8x32_epi32(v[0],i2),
           _mm256_permutevar8x32_epi32(v[1],i2),0x49),_mm256_permutevar8x32_epi32(v[2],i2),0x92);
}

/* Range of eight positions over 64 channels. Without NaN, min/max order
 * cannot change the result: the final comparison with +0 settles signed
 * zeros, as in quantize_range. A NaN group repeats that exact row order. */
static inline __attribute__((always_inline)) void cm_range(const float *x,int width,__m256 *lo_out,__m256 *hi_out) {
    __m256 lo=_mm256_setzero_ps(),hi=lo,nan=lo;
    for (int k=0;k<64;k+=2) {
        __m256 a=_mm256_loadu_ps(x+k*width),b=_mm256_loadu_ps(x+(k+1)*width);
        lo=_mm256_min_ps(_mm256_min_ps(lo,a),b); hi=_mm256_max_ps(_mm256_max_ps(hi,a),b);
        nan=_mm256_or_ps(nan,_mm256_cmp_ps(a,b,_CMP_UNORD_Q));
    }
    if (__builtin_expect(_mm256_movemask_ps(nan),0)) {
        float low[8],high[8],column[64];
        for (int i=0;i<8;++i) {
            for (int k=0;k<64;++k) column[k]=x[k*width+i];
            quantize_range(column,64,low+i,high+i);
        }
        *lo_out=_mm256_loadu_ps(low); *hi_out=_mm256_loadu_ps(high);
        return;
    }
    *lo_out=_mm256_min_ps(lo,_mm256_setzero_ps());
    *hi_out=_mm256_max_ps(hi,_mm256_setzero_ps());
}

/* Four channels of eight positions -> [position][4 channels] bytes. */
static inline __attribute__((always_inline)) __m256i cm_bytes(const float *x,int width,
        __m256 live,__m256 inv,__m256 z) {
    __m256i q[4];
    for (int c=0;c<4;++c) {
        /* Constant positions convert +0 with zero parameters: no new exceptions. */
        __m256 v=_mm256_and_ps(_mm256_loadu_ps(x+c*width),live);
        q[c]=_mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(v,inv),z));
    }
    __m256i p=_mm256_packus_epi16(_mm256_packs_epi32(q[0],q[1]),_mm256_packs_epi32(q[2],q[3]));
    p=_mm256_min_epu8(p,_mm256_set1_epi8((char)254));
    const __m256i order=_mm256_setr_epi8(0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15,
                                         0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15);
    return _mm256_shuffle_epi8(p,order);
}

/* Integer sums for four outputs (n0..n0+3) over DPDF_CM_VECTORS position
 * vectors. Inline assembly: GCC spills a 16-register intrinsic version on
 * every iteration. Operands are the accumulators; addresses and the K walk
 * are those of the intrinsic tail below. */
static inline __attribute__((always_inline)) void cm_dot(const uint8_t *act,int stride,
        const int8_t *w,__m256i sum[4][DPDF_CM_VECTORS]) {
#ifdef DPDF_INT8_VNNI
    __m256i s00,s01,s02,s10,s11,s12,s20,s21,s22,s30,s31,s32;
    __m256i a0,a1,a2,b;
    const uint8_t *end=act+16*(size_t)stride;
    __asm__(
        "vpxor %x[s00],%x[s00],%x[s00]\n\t"
        "vpxor %x[s01],%x[s01],%x[s01]\n\t"
        "vpxor %x[s02],%x[s02],%x[s02]\n\t"
        "vpxor %x[s10],%x[s10],%x[s10]\n\t"
        "vpxor %x[s11],%x[s11],%x[s11]\n\t"
        "vpxor %x[s12],%x[s12],%x[s12]\n\t"
        "vpxor %x[s20],%x[s20],%x[s20]\n\t"
        "vpxor %x[s21],%x[s21],%x[s21]\n\t"
        "vpxor %x[s22],%x[s22],%x[s22]\n\t"
        "vpxor %x[s30],%x[s30],%x[s30]\n\t"
        "vpxor %x[s31],%x[s31],%x[s31]\n\t"
        "vpxor %x[s32],%x[s32],%x[s32]\n\t"
        "1:\n\t"
        "vmovdqa 0(%[act]),%t[a0]\n\t"
        "vmovdqa 32(%[act]),%t[a1]\n\t"
        "vmovdqa 64(%[act]),%t[a2]\n\t"
        "vpbroadcastd 0(%[w]),%t[b]\n\t"
        "%{vex%} vpdpbusd %t[b],%t[a0],%t[s00]\n\t"
        "%{vex%} vpdpbusd %t[b],%t[a1],%t[s01]\n\t"
        "%{vex%} vpdpbusd %t[b],%t[a2],%t[s02]\n\t"
        "vpbroadcastd 4(%[w]),%t[b]\n\t"
        "%{vex%} vpdpbusd %t[b],%t[a0],%t[s10]\n\t"
        "%{vex%} vpdpbusd %t[b],%t[a1],%t[s11]\n\t"
        "%{vex%} vpdpbusd %t[b],%t[a2],%t[s12]\n\t"
        "vpbroadcastd 8(%[w]),%t[b]\n\t"
        "%{vex%} vpdpbusd %t[b],%t[a0],%t[s20]\n\t"
        "%{vex%} vpdpbusd %t[b],%t[a1],%t[s21]\n\t"
        "%{vex%} vpdpbusd %t[b],%t[a2],%t[s22]\n\t"
        "vpbroadcastd 12(%[w]),%t[b]\n\t"
        "%{vex%} vpdpbusd %t[b],%t[a0],%t[s30]\n\t"
        "%{vex%} vpdpbusd %t[b],%t[a1],%t[s31]\n\t"
        "%{vex%} vpdpbusd %t[b],%t[a2],%t[s32]\n\t"
        "add %[stride],%[act]\n\t"
        "add $32,%[w]\n\t"
        "cmp %[end],%[act]\n\t"
        "jne 1b"
        : [s00]"=&x"(s00), [s01]"=&x"(s01), [s02]"=&x"(s02), [s10]"=&x"(s10), [s11]"=&x"(s11), [s12]"=&x"(s12), [s20]"=&x"(s20), [s21]"=&x"(s21), [s22]"=&x"(s22), [s30]"=&x"(s30), [s31]"=&x"(s31), [s32]"=&x"(s32), [a0]"=&x"(a0), [a1]"=&x"(a1), [a2]"=&x"(a2), [b]"=&x"(b), [act]"+r"(act), [w]"+r"(w)
        : [stride]"r"((size_t)stride), [end]"r"(end)
        : "cc");
    sum[0][0]=s00;
    sum[0][1]=s01;
    sum[0][2]=s02;
    sum[1][0]=s10;
    sum[1][1]=s11;
    sum[1][2]=s12;
    sum[2][0]=s20;
    sum[2][1]=s21;
    sum[2][2]=s22;
    sum[3][0]=s30;
    sum[3][1]=s31;
    sum[3][2]=s32;
#else
    const __m256i ones=_mm256_set1_epi16(1);
    __m256i s00,s01,s10,s11,s20,s21,s30,s31;
    __m256i a0,a1,b,t;
    const uint8_t *end=act+16*(size_t)stride;
    __asm__(
        "vpxor %x[s00],%x[s00],%x[s00]\n\t"
        "vpxor %x[s01],%x[s01],%x[s01]\n\t"
        "vpxor %x[s10],%x[s10],%x[s10]\n\t"
        "vpxor %x[s11],%x[s11],%x[s11]\n\t"
        "vpxor %x[s20],%x[s20],%x[s20]\n\t"
        "vpxor %x[s21],%x[s21],%x[s21]\n\t"
        "vpxor %x[s30],%x[s30],%x[s30]\n\t"
        "vpxor %x[s31],%x[s31],%x[s31]\n\t"
        "1:\n\t"
        "vmovdqa 0(%[act]),%t[a0]\n\t"
        "vmovdqa 32(%[act]),%t[a1]\n\t"
        "vpbroadcastd 0(%[w]),%t[b]\n\t"
        "vpmaddubsw %t[b],%t[a0],%t[t]\n\t"
        "vpmaddwd %t[ones],%t[t],%t[t]\n\t"
        "vpaddd %t[t],%t[s00],%t[s00]\n\t"
        "vpmaddubsw %t[b],%t[a1],%t[t]\n\t"
        "vpmaddwd %t[ones],%t[t],%t[t]\n\t"
        "vpaddd %t[t],%t[s01],%t[s01]\n\t"
        "vpbroadcastd 4(%[w]),%t[b]\n\t"
        "vpmaddubsw %t[b],%t[a0],%t[t]\n\t"
        "vpmaddwd %t[ones],%t[t],%t[t]\n\t"
        "vpaddd %t[t],%t[s10],%t[s10]\n\t"
        "vpmaddubsw %t[b],%t[a1],%t[t]\n\t"
        "vpmaddwd %t[ones],%t[t],%t[t]\n\t"
        "vpaddd %t[t],%t[s11],%t[s11]\n\t"
        "vpbroadcastd 8(%[w]),%t[b]\n\t"
        "vpmaddubsw %t[b],%t[a0],%t[t]\n\t"
        "vpmaddwd %t[ones],%t[t],%t[t]\n\t"
        "vpaddd %t[t],%t[s20],%t[s20]\n\t"
        "vpmaddubsw %t[b],%t[a1],%t[t]\n\t"
        "vpmaddwd %t[ones],%t[t],%t[t]\n\t"
        "vpaddd %t[t],%t[s21],%t[s21]\n\t"
        "vpbroadcastd 12(%[w]),%t[b]\n\t"
        "vpmaddubsw %t[b],%t[a0],%t[t]\n\t"
        "vpmaddwd %t[ones],%t[t],%t[t]\n\t"
        "vpaddd %t[t],%t[s30],%t[s30]\n\t"
        "vpmaddubsw %t[b],%t[a1],%t[t]\n\t"
        "vpmaddwd %t[ones],%t[t],%t[t]\n\t"
        "vpaddd %t[t],%t[s31],%t[s31]\n\t"
        "add %[stride],%[act]\n\t"
        "add $32,%[w]\n\t"
        "cmp %[end],%[act]\n\t"
        "jne 1b"
        : [s00]"=&x"(s00), [s01]"=&x"(s01), [s10]"=&x"(s10), [s11]"=&x"(s11), [s20]"=&x"(s20), [s21]"=&x"(s21), [s30]"=&x"(s30), [s31]"=&x"(s31), [a0]"=&x"(a0), [a1]"=&x"(a1), [b]"=&x"(b), [t]"=&x"(t), [act]"+r"(act), [w]"+r"(w)
        : [stride]"r"((size_t)stride), [end]"r"(end), [ones]"x"(ones)
        : "cc");
    sum[0][0]=s00;
    sum[0][1]=s01;
    sum[1][0]=s10;
    sum[1][1]=s11;
    sum[2][0]=s20;
    sum[2][1]=s21;
    sum[3][0]=s30;
    sum[3][1]=s31;
#endif
}

/* One vector: the position tail when the count is not a block multiple. */
static inline __attribute__((always_inline)) void cm_dot1(const uint8_t *act,int stride,
        const int8_t *w,__m256i sum[4][DPDF_CM_VECTORS]) {
    __m256i s0=_mm256_setzero_si256(),s1=s0,s2=s0,s3=s0;
    for (int g=0;g<16;++g) {
        __m256i a=_mm256_load_si256((const __m256i *)(act+g*stride));
        s0=DPDF_DOT4(s0,a,_mm256_set1_epi32(*(const int32_t *)(const void *)(w+g*32)));
        s1=DPDF_DOT4(s1,a,_mm256_set1_epi32(*(const int32_t *)(const void *)(w+g*32+4)));
        s2=DPDF_DOT4(s2,a,_mm256_set1_epi32(*(const int32_t *)(const void *)(w+g*32+8)));
        s3=DPDF_DOT4(s3,a,_mm256_set1_epi32(*(const int32_t *)(const void *)(w+g*32+12)));
    }
    sum[0][0]=s0; sum[1][0]=s1; sum[2][0]=s2; sum[3][0]=s3;
}

static inline __attribute__((always_inline)) void cm_block(const dpdf_qmatrix *q,
        const uint8_t *act,const float *scales,const int32_t *negzp,int positions,int vector,
        int n0,const float *bias,float *y,int relu,const int vectors) {
    __m256i sum[4][DPDF_CM_VECTORS];
    const int8_t *w=q->packed+(n0/8)*64*8+(n0%8)*4;
    if (vectors==1) cm_dot1(act+vector*32,positions*4,w,sum);
    else cm_dot(act+vector*32,positions*4,w,sum);
    for (int i=0;i<4;++i) {
        const int n=n0+i;
        const __m256i weight_sum=_mm256_set1_epi32(q->sum[n]);
        const __m256 weight_scale=_mm256_set1_ps(q->scale[n]),b=_mm256_set1_ps(bias[n]);
        for (int j=0;j<vectors;++j) {
            const int p=(vector+j)*8;
            __m256i corrected=_mm256_add_epi32(sum[i][j],
                _mm256_mullo_epi32(weight_sum,_mm256_loadu_si256((const __m256i *)(negzp+p))));
            __m256 scale=_mm256_mul_ps(_mm256_loadu_ps(scales+p),weight_scale);
            __m256 result=_mm256_fmadd_ps(_mm256_cvtepi32_ps(corrected),scale,b);
            if (relu) result=_mm256_max_ps(result,_mm256_setzero_ps());
            _mm256_storeu_ps(y+n*positions+p,result);
        }
    }
}

void DPDF_QCM_IMPL(const dpdf_qmatrix *q,const float *const *pieces,int count,int width,
                   const float *bias,float *y,int relu) {
    const int positions=count*width;
    _Alignas(32) uint8_t act[16*DPDF_CM_MAX*4];
    _Alignas(32) float scales[DPDF_CM_MAX];
    _Alignas(32) int32_t negzp[DPDF_CM_MAX];
    for (int t=0;t<width;t+=8) {
        __m256 inv[3],z[3],live[3];
        __m256i scale_bits[3],negative_zp[3];
        for (int j=0;j<count;++j) {
            __m256 lo,hi,scale,constant; __m256i zp;
            cm_range(pieces[j]+t,width,&lo,&hi);
            quantize_params8(lo,hi,&scale,&zp,inv+j,&constant);
            live[j]=_mm256_xor_ps(constant,_mm256_castsi256_ps(_mm256_set1_epi32(-1)));
            inv[j]=_mm256_and_ps(inv[j],live[j]);
            z[j]=_mm256_cvtepi32_ps(zp);
            scale_bits[j]=_mm256_castps_si256(scale);
            negative_zp[j]=_mm256_sub_epi32(_mm256_setzero_si256(),zp);
        }
        __m256i out[3];
        interleave32(scale_bits,count,out);
        for (int j=0;j<count;++j) _mm256_store_si256((__m256i *)(scales+t*count+j*8),out[j]);
        interleave32(negative_zp,count,out);
        for (int j=0;j<count;++j) _mm256_store_si256((__m256i *)(negzp+t*count+j*8),out[j]);
        for (int g=0;g<16;++g) {
            __m256i bytes[3];
            for (int j=0;j<count;++j) bytes[j]=cm_bytes(pieces[j]+g*4*width+t,width,live[j],inv[j],z[j]);
            interleave32(bytes,count,out);
            for (int j=0;j<count;++j) _mm256_store_si256((__m256i *)(act+g*positions*4+(t*count+j*8)*4),out[j]);
        }
    }
    const int vectors=positions/8;
    for (int n0=0;n0<64;n0+=4) {
        int v=0;
        for (;v+DPDF_CM_VECTORS<=vectors;v+=DPDF_CM_VECTORS)
            cm_block(q,act,scales,negzp,positions,v,n0,bias,y,relu,DPDF_CM_VECTORS);
        for (;v<vectors;++v) cm_block(q,act,scales,negzp,positions,v,n0,bias,y,relu,1);
    }
}
#undef DPDF_CM_VECTORS

#undef DPDF_DOT4
