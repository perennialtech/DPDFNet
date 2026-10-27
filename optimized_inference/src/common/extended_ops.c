#include "extended_ops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct dpdf_dense {
    int k,n,kp,np;
    float *bias,*x,*y;
    dpdf_qmatrix *q;
};
void dpdf_dense_destroy(dpdf_dense *d) {
    if (!d) return;
    free(d->bias); free(d->x); free(d->y);
    dpdf_qdestroy(d->q);
    free(d);
}
dpdf_dense *dpdf_dense_create(const float *w,const float *bias,int k,int n,int tiled) {
    if (!w || k<=0 || k>512 || n<=0 || n>1024) return NULL;
    if (tiled && (tiled!=64 || n%64)) return NULL;
    dpdf_dense *d=calloc(1,sizeof(*d)); if (!d) return NULL;
    d->k=k; d->n=n; d->kp=(k+7)/8*8; d->np=(n+7)/8*8;
    float *packed=calloc((size_t)d->kp*d->np,sizeof(float));
    d->bias=calloc(d->np,sizeof(float));
    if (d->k!=d->kp) d->x=calloc((size_t)48*d->kp,sizeof(float));
    if (d->n!=d->np) d->y=malloc((size_t)48*d->np*sizeof(float));
    if (!packed || !d->bias || (d->k!=d->kp && !d->x) || (d->n!=d->np && !d->y)) {
        free(packed); dpdf_dense_destroy(d); return NULL;
    }
    for (int j=0;j<k;++j) for (int c=0;c<n;++c) {
        float v=w[tiled?(c/64)*k*64+j*64+c%64:j*n+c];
        if (!isfinite(v)) { free(packed); dpdf_dense_destroy(d); return NULL; }
        packed[j*d->np+c]=v;
    }
    if (bias) memcpy(d->bias,bias,n*sizeof(float));
    d->q=dpdf_qcreate(packed,d->kp,d->np);
    free(packed);
    if (!d->q) { dpdf_dense_destroy(d); return NULL; }
    return d;
}
void dpdf_dense_run(dpdf_dense *d,const float *x,float *y,int rows) {
    for (int start=0;start<rows;start+=48) {
        int m=rows-start; if (m>48) m=48;
        const float *input=x+start*d->k;
        if (d->k!=d->kp) {
            for (int r=0;r<m;++r) memcpy(d->x+r*d->kp,input+r*d->k,d->k*sizeof(float));
            input=d->x;
        }
        float *output=d->np==d->n ? y+start*d->n : d->y;
        dpdf_qaffine(d->q,input,d->bias,output,m);
        if (output==d->y) for (int r=0;r<m;++r) memcpy(y+(start+r)*d->n,output+r*d->np,d->n*sizeof(float));
    }
}
size_t dpdf_dense_bytes(const dpdf_dense *d) {
    if (!d) return 0;
    size_t b=sizeof(*d)+(size_t)d->np*4;
    if (d->x) b+=(size_t)48*d->kp*4;
    if (d->y) b+=(size_t)48*d->np*4;
    return b+dpdf_qbytes(d->q);
}

struct dpdf_convop {
    int ci,hi,wi,co,ho,wo,kh,kw,sh,sw,ph,pw,group;
    float *w,*bias;
};
void dpdf_convop_destroy(dpdf_convop *c) {
    if (!c) return;
    free(c->w); free(c->bias); free(c);
}
dpdf_convop *dpdf_convop_create(const float *w,const float *bias,int ci,int hi,int wi,int co,int ho,int wo,
 int kh,int kw,int sh,int sw,int ph,int pw,int group) {
    dpdf_convop *c=calloc(1,sizeof(*c)); if (!c) return NULL;
    c->ci=ci;c->hi=hi;c->wi=wi;c->co=co;c->ho=ho;c->wo=wo;c->kh=kh;c->kw=kw;
    c->sh=sh;c->sw=sw;c->ph=ph;c->pw=pw;c->group=group;
    int k=ci/group;
    k*=kh*kw;
    c->w=malloc((size_t)co*k*sizeof(float));c->bias=calloc(co,sizeof(float));
    if (!c->w || !c->bias) { dpdf_convop_destroy(c); return NULL; }
    memcpy(c->w,w,(size_t)co*k*sizeof(float));
    if (bias) memcpy(c->bias,bias,co*sizeof(float));
    return c;
}
void dpdf_convop_run(dpdf_convop *c,const float *x,float *y) {
    dpdf_conv(dpdf_axpy_avx2,x,c->w,c->bias,y,c->ci,c->hi,c->wi,c->co,c->ho,c->wo,
              c->kh,c->kw,c->sh,c->sw,c->ph,c->pw,c->group);
}
size_t dpdf_convop_bytes(const dpdf_convop *c) {
    if (!c) return 0;
    size_t k=(size_t)c->ci/c->group*c->kh*c->kw;
    return sizeof(*c)+c->co*k*4+c->co*4;
}
