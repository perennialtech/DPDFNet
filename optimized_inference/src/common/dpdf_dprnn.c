#include "internal.h"
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

struct dpdf_block {
    int freq;
    float eps[2];
    dpdf_qmatrix *q[8];
    float params[1536]; /* Bias and normalization parameters remain FP32. */
    dpdf_gates_fn gates;
    dpdf_norm_fn norm;
};

int dpdf_has_avx2(void) {
    __builtin_cpu_init();
    return !!(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"));
}

static const int matrix_offset[8]={0,12288,24576,36864,49920,58304,70592,83264};

dpdf_block *dpdf_create_int8(int f,const float *w,size_t count,float e0,float e1) {
    if ((f!=40 && f!=48) || !w || count!=DPDF_WEIGHT_FLOATS ||
        !dpdf_has_avx2() || !isfinite(e0) || !isfinite(e1) || e0<=0 || e1<=0)
        return NULL;
    for (size_t i=0;i<count;++i) if (!isfinite(w[i])) return NULL;
    dpdf_block *b=calloc(1,sizeof(*b));
    if (!b) return NULL;
    b->freq=f; b->eps[0]=e0; b->eps[1]=e1;
    b->gates=dpdf_gates_avx2; b->norm=dpdf_norm_residual_avx2;
    memcpy(b->params,w+49152,768*sizeof(float));
    memcpy(b->params+768,w+58112,192*sizeof(float));
    memcpy(b->params+960,w+82880,384*sizeof(float));
    memcpy(b->params+1344,w+87360,192*sizeof(float));
    for (int i=0;i<8;++i) {
        int k=i==4?128:64,n=(i==4 || i==7)?64:192;
        b->q[i]=dpdf_qcreate(w+matrix_offset[i],k,n);
        if (!b->q[i]) { dpdf_destroy(b); return NULL; }
    }
    return b;
}
void dpdf_destroy(dpdf_block *b) {
    if (!b) return;
    for (int i=0;i<8;++i) dpdf_qdestroy(b->q[i]);
    free(b);
}
size_t dpdf_block_bytes(const dpdf_block *b) {
    if (!b) return 0;
    size_t bytes=sizeof(*b);
    for (int i=0;i<8;++i) bytes+=dpdf_qbytes(b->q[i]);
    return bytes;
}
static inline void affine(const dpdf_block *b,int matrix,const float *x,const float *bias,
                          float *y,int rows) {
    dpdf_qaffine(b->q[matrix],x,bias,y,rows);
}

int dpdf_process_layout(const dpdf_block *b, const float *input, const float *state,
                        float *output, float *state_out, unsigned layout_flags) {
    if (!b || !input || !state || !output || !state_out || layout_flags > 3) return -1;
    const int f = b->freq;
    float x[48*64], bi[48*128], mid[48*64], a[48*192], hproj[48*192], p[48*64];
    float h[64], step[192];
    const float *bias=b->params,*fib=b->params+768,*nis=fib+64,*nib=nis+64;
    const float *tib=b->params+960,*thb=tib+192,*fob=b->params+1344,*nos=fob+64,*nob=nos+64;
    const float *features=(layout_flags & DPDF_INPUT_FREQ_MAJOR)?input:x;
    if (!(layout_flags & DPDF_INPUT_FREQ_MAJOR)) {
#if DPDF_MODEL_SIZE == 2
        dpdf_transpose_avx2(input,x,64,f);
#else
        for (int r=0;r<f;++r) for (int c=0;c<64;++c) x[r*64+c]=input[c*f+r];
#endif
    }
    float *intra[2]={a,hproj};
    dpdf_qaffine_pair(b->q[0],b->q[1],features,bias,bias+384,a,hproj,f);
    for (int direction = 0; direction < 2; ++direction) {
        memset(h, 0, sizeof(h));

        for (int t = 0; t < f; ++t) {
            int r = direction == 0 ? t : f-1-t;
            if (t==0) {
                const float *initial_bias=bias+direction*384+192;
                for (int c=0;c<192;++c) step[c]=0.0f+initial_bias[c];
            } else
                affine(b, 2+direction, h, bias+direction*384+192, step, 1);
            b->gates(intra[direction]+r*192, step, h, h, 1);
            memcpy(bi+r*128+direction*64, h, sizeof(h));
        }
    }
    affine(b, 4, bi, fib, p, f);
    b->norm(mid, p, features, nis, nib, b->eps[0], f);
    affine(b, 5, mid, tib, a, f);
    affine(b, 6, state, thb, hproj, f);
    b->gates(a, hproj, state, state_out, f);
    affine(b, 7, state_out, fob, p, f);
    float *final=(layout_flags & DPDF_OUTPUT_FREQ_MAJOR)?output:x;
    b->norm(final, p, mid, nos, nob, b->eps[1], f);
    if (!(layout_flags & DPDF_OUTPUT_FREQ_MAJOR)) {
#if DPDF_MODEL_SIZE == 2
        dpdf_transpose_avx2(x,output,f,64);
#else
        for (int r=0;r<f;++r) for (int c=0;c<64;++c) output[c*f+r]=x[r*64+c];
#endif
    }
    return 0;
}
