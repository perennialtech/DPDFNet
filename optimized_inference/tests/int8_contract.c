/* Independent scalar integer arithmetic checks the SIMD dot path, including
 * extreme signed values that would expose saturated pair sums. */
#include "internal.h"
#include "int8_matrix.h"
#include <fenv.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"INT8 test line %d\n",__LINE__); return 1; } } while (0)

static uint32_t rng=0x194812;
static float sample(void) {
    rng=rng*1664525u+1013904223u;
    return ((int)(rng>>16)-32768)/8192.0f;
}

/* Independent scalar quantization/dot oracle, including M=1, short widths,
 * 64-output tile tails and buffers without 32-byte alignment. */
static int row_contract(void) {
    const int widths[]={8,56,64,72,128,192,768};
    const int depths[]={8,24,64,128,256,512};
    for (size_t ki=0;ki<sizeof(depths)/sizeof(depths[0]);++ki)
      for (size_t ni=0;ni<sizeof(widths)/sizeof(widths[0]);++ni) {
        int k=depths[ki],n=widths[ni];
        float *w=malloc((size_t)k*n*sizeof(float));
        float *xa=malloc((k+1)*sizeof(float)),*ya=malloc((n+2)*sizeof(float));
        float *ba=malloc((n+1)*sizeof(float));
        CHECK(w && xa && ya && ba);
        float *x=xa+1,*y=ya+1,*bias=ba+1;
        for (int j=0;j<k;++j) for (int c=0;c<n;++c)
            w[j*n+c]=c%5==0 ? 0 : c%5==1 ? (j%2 ? -1.0f : 1.0f) : sample();
        for (int c=0;c<n;++c) bias[c]=sample();
        dpdf_qmatrix *q=dpdf_qcreate(w,k,n); CHECK(q);
        for (int pattern=0;pattern<8;++pattern) {
            for (int j=0;j<k;++j)
                x[j]=pattern==0 ? 0 : pattern==1 ? .5f : pattern==2 ? -.5f :
                     pattern==3 ? (j%2 ? -1.0f : 1.0f) : sample();
            float low=0,high=0;
            for (int j=0;j<k;++j) { low=fminf(low,x[j]); high=fmaxf(high,x[j]); }
            float scale=low==high ? 1 : (high-low)/254.0f;
            int zp=low==high ? 127 : (int)nearbyintf(-low/scale);
            if (zp<0) zp=0;
            if (zp>254) zp=254;
            int aq[512];
            for (int j=0;j<k;++j) {
                int v=(int)nearbyintf(x[j]*(1.0f/scale)+(float)zp);
                if (v<0) v=0;
                if (v>254) v=254;
                aq[j]=v-zp;
            }
            ya[0]=12345; ya[n+1]=-12345;
            dpdf_qaffine(q,x,bias,y,1);
            CHECK(ya[0]==12345 && ya[n+1]==-12345);
            for (int c=0;c<n;++c) {
                float maximum=0;
                for (int j=0;j<k;++j) maximum=fmaxf(maximum,fabsf(w[j*n+c]));
                float ws=maximum>0 ? maximum/63.0f : 1;
                int32_t sum=0;
                for (int j=0;j<k;++j) {
                    int v=(int)nearbyintf(w[j*n+c]/ws);
                    if (v>63) v=63;
                    if (v< -63) v= -63;
                    sum+=aq[j]*v;
                }
                float expected=fmaf((float)sum,scale*ws,bias[c]);
                CHECK(memcmp(y+c,&expected,sizeof(float))==0);
            }
        }
        dpdf_qdestroy(q); free(w); free(xa); free(ya); free(ba);
      }
    return 0;
}

static const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};

/* Row r of a batch: special rows (zeros of both signs, a lone value,
 * denormals, huge magnitudes) between ordinary ones. */
static float batch_value(int r,int j) {
    float v=sample();
    switch (r%8) {
    case 0: return 0.0f;
    case 1: return (j%3) ? -0.0f : 0.0f;
    case 2: return j==5 ? v : 0.0f;
    case 3: return v*1e-39f;
    case 4: return (j%2) ? 3e37f : -3e37f;
    case 5: return v*1e-3f;
    default: return v;
    }
}

/* Batch quantization derives eight rows' parameters per vector; each row
 * must match the scalar M=1 path in every rounding mode, with and without
 * FTZ/DAZ. */
static int batch_contract(void) {
    const int depths[]={8,64,128,512},widths[]={8,64,72,192},rows[]={4,7,40,48};
    fenv_t saved; CHECK(fegetenv(&saved)==0);
    const unsigned original_mxcsr=_mm_getcsr();
    for (int ki=0;ki<4;++ki) for (int ni=0;ni<4;++ni) {
        int k=depths[ki],n=widths[ni];
        float *w=malloc((size_t)k*n*4),*x=malloc((size_t)48*k*4),*bias=malloc(n*4);
        float *y=malloc((size_t)(48*n+2)*4),*z=malloc((size_t)(48*n+2)*4),*ref=malloc(n*4);
        CHECK(w && x && bias && y && z && ref);
        for (int i=0;i<k*n;++i) w[i]=sample();
        dpdf_qmatrix *q0=dpdf_qcreate(w,k,n); CHECK(q0);
        for (int i=0;i<k*n;++i) w[i]=sample();
        dpdf_qmatrix *q1=dpdf_qcreate(w,k,n); CHECK(q1);
        for (int i=0;i<48*k;++i) x[i]=batch_value(i/k,i%k);
        for (int c=0;c<n;++c) bias[c]=sample();
        for (int mode=0;mode<8;++mode) {
            _mm_setcsr((original_mxcsr&~0x8040u)|((mode&4)?0x8040u:0u));
            CHECK(fesetround(modes[mode&3])==0);
            for (int mi=0;mi<4;++mi) {
                int m=rows[mi];
                y[0]=z[0]=12345;y[m*n+1]=z[m*n+1]=-12345;
                dpdf_qaffine(q0,x,bias,y+1,m);
                for (int r=0;r<m;++r) {
                    dpdf_qaffine(q0,x+r*k,bias,ref,1);
                    CHECK(!memcmp(y+1+r*n,ref,n*4));
                }
                dpdf_qaffine_pair(q0,q1,x,bias,bias,y+1,z+1,m);
                for (int r=0;r<m;++r) {
                    dpdf_qaffine(q0,x+r*k,bias,ref,1); CHECK(!memcmp(y+1+r*n,ref,n*4));
                    dpdf_qaffine(q1,x+r*k,bias,ref,1); CHECK(!memcmp(z+1+r*n,ref,n*4));
                }
                CHECK(y[0]==12345 && z[0]==12345 && y[m*n+1]==-12345 && z[m*n+1]==-12345);
            }
        }
        dpdf_qdestroy(q0);dpdf_qdestroy(q1);
        free(w);free(x);free(bias);free(y);free(z);free(ref);
    }
    CHECK(fesetenv(&saved)==0); _mm_setcsr(original_mxcsr);
    return 0;
}

/* The pipelined bidirectional recurrence against the sequential loop it
 * replaced: per direction, h starts at zero, the first step is 0+bias and
 * each later step is the M=1 projection of h, then the gates. */
static void recur_oracle(const dpdf_qmatrix *q0,const dpdf_qmatrix *q1,const float *intra0,
        const float *intra1,const float *bias0,const float *bias1,int f,float *bi) {
    const dpdf_qmatrix *q[2]={q0,q1};
    const float *intra[2]={intra0,intra1},*bias[2]={bias0,bias1};
    float h[64],step[192];
    for (int direction=0;direction<2;++direction) {
        memset(h,0,sizeof(h));
        for (int t=0;t<f;++t) {
            int r=direction==0 ? t : f-1-t;
            if (t==0) for (int c=0;c<192;++c) step[c]=0.0f+bias[direction][c];
            else dpdf_qaffine(q[direction],h,bias[direction],step,1);
            dpdf_gates_avx2(intra[direction]+r*192,step,h,h,1);
            memcpy(bi+r*128+direction*64,h,sizeof(h));
        }
    }
}

static int recur_contract(void) {
    const int lengths[]={1,2,3,40,48};
    float *w=malloc(64*192*4),*intra=malloc(2*48*192*4),*bias=malloc(2*192*4);
    float *expected=malloc(48*128*4),*actual=malloc(48*128*4);
    CHECK(w && intra && bias && expected && actual);
    for (int i=0;i<64*192;++i) w[i]=sample()*0.25f;
    dpdf_qmatrix *q0=dpdf_qcreate(w,64,192); CHECK(q0);
    for (int i=0;i<64*192;++i) w[i]=sample()*0.25f;
    dpdf_qmatrix *q1=dpdf_qcreate(w,64,192); CHECK(q1);
    for (int i=0;i<2*48*192;++i) intra[i]=sample();
    for (int i=0;i<2*192;++i) bias[i]=sample()*0.5f;
    dpdf_qrecur_impl impls[2]={dpdf_qrecur_avx2,NULL};
#ifdef DPDF_HAS_AVX_VNNI
    if (dpdf_has_avx_vnni()) impls[1]=dpdf_qrecur_vnni;
#endif
    fenv_t saved; CHECK(fegetenv(&saved)==0);
    const unsigned original_mxcsr=_mm_getcsr();
    for (int mode=0;mode<8;++mode) {
        _mm_setcsr((original_mxcsr&~0x8040u)|((mode&4)?0x8040u:0u));
        CHECK(fesetround(modes[mode&3])==0);
        for (size_t li=0;li<sizeof(lengths)/sizeof(lengths[0]);++li) {
            const int f=lengths[li];
            recur_oracle(q0,q1,intra,intra+48*192,bias,bias+192,f,expected);
            for (int impl=0;impl<2;++impl) {
                if (!impls[impl]) continue;
                memset(actual,0xff,(size_t)f*128*4);
                impls[impl](q0,q1,intra,intra+48*192,bias,bias+192,f,actual);
                CHECK(!memcmp(actual,expected,(size_t)f*128*4));
            }
        }
    }
    CHECK(fesetenv(&saved)==0); _mm_setcsr(original_mxcsr);
    dpdf_qdestroy(q0); dpdf_qdestroy(q1);
    free(w);free(intra);free(bias);free(expected);free(actual);
    return 0;
}

int main(void) {
    if (!dpdf_has_avx2()) { puts("INT8 SIMD unavailable; skipped"); return 0; }
    CHECK(row_contract()==0);
    CHECK(batch_contract()==0);
    CHECK(recur_contract()==0);
    puts("Reduced-range scalar oracle, batch and recurrence contracts passed");
    return 0;
}
