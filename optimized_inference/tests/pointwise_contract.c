/* Channel-major pointwise projection against the row-major W7A8 path.
 * The reference gathers each position into a row, runs the single-row
 * affine path, transposes back and applies the graph's x>0?x:0. Covers
 * every piece count and width the graphs use, edge widths, special values
 * (including NaN groups that take the exact row-order range), in-place
 * output, both ISA implementations and every rounding mode with and
 * without FTZ/DAZ. */
#include "int8_matrix.h"
#include "extended_ops.h"
#include <fenv.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"Pointwise test line %d\n",__LINE__); return 1; } } while (0)

static uint32_t seed=0x5eed1234u;
static uint32_t next(void) { seed=seed*1664525u+1013904223u; return seed; }
static float sample(void) { return ((int)(next()>>16)-32768)/8192.0f; }
static float bits(uint32_t value) { float f; memcpy(&f,&value,4); return f; }

/* Column t of piece j; pattern selects the special values exercised. */
static float value(int pattern,int j,int k,int t) {
    float v=sample();
    switch (pattern) {
    case 1: return (t%5==0) ? 0.0f : v;                          /* constant positions */
    case 2: return (t%3==0) ? bits(next()&0x80000000u) : v;        /* signed zeros only */
    case 3: return (t%7==1 && k==(j*5+t)%64) ? bits(0x7fc00000u|(next()&0xffu)) : v; /* NaN */
    case 4: return (t%6==2 && k==9) ? (next()&1 ? INFINITY : -INFINITY) : v;
    case 5: return (t%4==3) ? (k%2 ? 1e30f : -1e30f) : v*1e-3f;
    case 6: return (t%2) ? bits(next()&0x807fffffu) : v;           /* denormals */
    default: return v;
    }
}

/* One row at a time: M=1 takes the scalar per-row quantization parameters,
 * not the eight-row batch derivation the channel-major kernel shares. */
static void reference(const dpdf_qmatrix *q,const float *pieces,int count,int width,
                      const float *bias,float *y,int relu) {
    const int positions=count*width;
    float row[64],out[64];
    for (int p=0;p<positions;++p) {
        const int t=p/count,j=p%count;
        for (int k=0;k<64;++k) row[k]=pieces[j*64*width+k*width+t];
        dpdf_qaffine_avx2(q,row,bias,out,1);
        for (int n=0;n<64;++n) {
            float v=out[n];
            y[n*positions+p]=relu ? (v>0?v:0.0f) : v;
        }
    }
}

int main(void) {
    const int widths[]={8,16,24,40,48,80,96,160};
    const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
    float *w=malloc(64*64*sizeof(float)),*bias=malloc(64*sizeof(float));
    float *input=malloc(480*64*sizeof(float)),*expected=malloc(480*64*sizeof(float));
    float *storage=malloc((480*64+2)*sizeof(float)),*inplace=malloc(480*64*sizeof(float));
    CHECK(w && bias && input && expected && storage && inplace);
    for (int i=0;i<64*64;++i) w[i]=sample();
    for (int i=0;i<64;++i) bias[i]=sample();
    dpdf_qmatrix *q=dpdf_qcreate(w,64,64); CHECK(q);
    dpdf_qcm_impl impls[2]={dpdf_qaffine_cm_avx2,NULL};
#ifdef DPDF_HAS_AVX_VNNI
    if (dpdf_has_avx_vnni()) impls[1]=dpdf_qaffine_cm_vnni;
#endif
    CHECK(q->cm==(impls[1] ? impls[1] : impls[0]));
    fenv_t saved; CHECK(fegetenv(&saved)==0);
    const unsigned original_mxcsr=_mm_getcsr();
    int cases=0;
    for (int count=1;count<=3;++count) for (size_t wi=0;wi<sizeof(widths)/sizeof(widths[0]);++wi) {
        const int width=widths[wi],positions=count*width,total=64*positions;
        if (positions>480) continue;
        for (int pattern=0;pattern<7;++pattern) for (int relu=0;relu<2;++relu) {
            for (int j=0;j<count;++j) for (int k=0;k<64;++k) for (int t=0;t<width;++t)
                input[j*64*width+k*width+t]=value(pattern,j,k,t);
            const float *pieces[3]={input,input+64*width,input+128*width};
            for (int mode=0;mode<8;++mode) {
                _mm_setcsr((original_mxcsr&~0x8040u)|((mode&4)?0x8040u:0u));
                CHECK(fesetround(modes[mode&3])==0);
                const unsigned controls=_mm_getcsr()&0xffc0u;
                reference(q,input,count,width,bias,expected,relu);
                for (int impl=0;impl<2;++impl) {
                    if (!impls[impl]) continue;
                    float *y=storage+1;
                    storage[0]=12345.0f; storage[total+1]=-12345.0f;
                    impls[impl](q,pieces,count,width,bias,y,relu);
                    CHECK(storage[0]==12345.0f && storage[total+1]==-12345.0f);
                    CHECK(!memcmp(y,expected,(size_t)total*sizeof(float)));
                    /* Output over the whole contiguous input. */
                    memcpy(inplace,input,(size_t)total*sizeof(float));
                    const float *same[3]={inplace,inplace+64*width,inplace+128*width};
                    impls[impl](q,same,count,width,bias,inplace,relu);
                    CHECK(!memcmp(inplace,expected,(size_t)total*sizeof(float)));
                    ++cases;
                }
                CHECK(fegetround()==modes[mode&3] && (_mm_getcsr()&0xffc0u)==controls);
            }
        }
    }
    CHECK(fesetenv(&saved)==0); _mm_setcsr(original_mxcsr);
    /* The dense wrapper accepts only the shapes the kernel handles. */
    dpdf_dense *d=dpdf_dense_create(w,bias,64,64,0),*narrow=dpdf_dense_create(w,bias,64,56,0);
    CHECK(d && narrow);
    const float *one[1]={input};
    CHECK(dpdf_dense_cm_run(d,one,1,8,storage,1)==0);
    CHECK(dpdf_dense_cm_run(d,one,1,12,storage,1)==-1);
    CHECK(dpdf_dense_cm_run(d,one,4,8,storage,1)==-1);
    CHECK(dpdf_dense_cm_run(d,one,3,168,storage,1)==-1);
    CHECK(dpdf_dense_cm_run(narrow,one,1,8,storage,1)==-1);
    dpdf_dense_destroy(d); dpdf_dense_destroy(narrow);
    dpdf_qdestroy(q);
    free(w);free(bias);free(input);free(expected);free(storage);free(inplace);
    printf("{\"passed\":true,\"cases\":%d}\n",cases);
    return 0;
}
