#ifndef DPDF_INT8_MATRIX_H
#define DPDF_INT8_MATRIX_H
#include "internal.h"

typedef void (*dpdf_qaffine_impl)(const dpdf_qmatrix *,const float *,const float *,float *,int);
typedef void (*dpdf_qpair_impl)(const dpdf_qmatrix *,const dpdf_qmatrix *,const float *,
                               const float *,const float *,float *,float *,int);
typedef void (*dpdf_qrecur_impl)(const dpdf_qmatrix *,const dpdf_qmatrix *,const float *,
                                 const float *,const float *,const float *,int,float *);
typedef void (*dpdf_qcm_impl)(const dpdf_qmatrix *,const float *const *,int,int,
                              const float *,float *,int);

struct dpdf_qmatrix {
    int k,n;
    int8_t *packed;
    void *packed_allocation;
    float *scale;
    int32_t *sum;
    dpdf_qaffine_impl affine;
    dpdf_qpair_impl pair;
    dpdf_qcm_impl cm;
    dpdf_qrecur_impl recur;
};

int dpdf_has_avx_vnni(void);
void dpdf_qaffine_avx2(const dpdf_qmatrix *,const float *,const float *,float *,int);
void dpdf_qaffine_pair_avx2(const dpdf_qmatrix *,const dpdf_qmatrix *,const float *,
                          const float *,const float *,float *,float *,int);
void dpdf_qaffine_cm_avx2(const dpdf_qmatrix *,const float *const *,int,int,const float *,float *,int);
void dpdf_qrecur_avx2(const dpdf_qmatrix *,const dpdf_qmatrix *,const float *,const float *,
                      const float *,const float *,int,float *);
#ifdef DPDF_HAS_AVX_VNNI
void dpdf_qaffine_vnni(const dpdf_qmatrix *,const float *,const float *,float *,int);
void dpdf_qaffine_pair_vnni(const dpdf_qmatrix *,const dpdf_qmatrix *,const float *,
                          const float *,const float *,float *,float *,int);
void dpdf_qaffine_cm_vnni(const dpdf_qmatrix *,const float *const *,int,int,const float *,float *,int);
void dpdf_qrecur_vnni(const dpdf_qmatrix *,const dpdf_qmatrix *,const float *,const float *,
                      const float *,const float *,int,float *);
#endif
#endif
