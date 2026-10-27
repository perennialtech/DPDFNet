#ifndef DPDF_INTERNAL_H
#define DPDF_INTERNAL_H
#include "dpdf_dprnn.h"
#include <stdint.h>
#ifdef DPDF_X86_DISPATCH
void dpdf_transpose_avx2(const float *,float *,int,int);
typedef struct dpdf_qmatrix dpdf_qmatrix;
dpdf_qmatrix *dpdf_qcreate(const float *, int, int);
void dpdf_qdestroy(dpdf_qmatrix *);
size_t dpdf_qbytes(const dpdf_qmatrix *);
void dpdf_qaffine(const dpdf_qmatrix *, const float *, const float *, float *, int);
void dpdf_qaffine_pair(const dpdf_qmatrix *, const dpdf_qmatrix *, const float *,
                       const float *, const float *, float *, float *, int);
void dpdf_gates_avx2(const float *, const float *, const float *, float *, int);
void dpdf_norm_residual_avx2(float *,const float *,const float *,const float *,const float *,float,int);
#endif
#endif
