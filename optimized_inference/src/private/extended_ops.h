#ifndef DPDF_EXTENDED_OPS_H
#define DPDF_EXTENDED_OPS_H
#include "full_ops.h"
typedef struct dpdf_dense dpdf_dense;
typedef struct dpdf_convop dpdf_convop;
/* Dense projections always use W7A8; convolutions here remain FP32.
 * Dense source is input-major; tiled=64 accepts [N/64,K,64] source packing. */
dpdf_dense *dpdf_dense_create(const float *,const float *,int,int,int);
void dpdf_dense_destroy(dpdf_dense *);
void dpdf_dense_run(dpdf_dense *,const float *,float *,int);
/* Pointwise convolution without layout passes: count (1-3) channel-major
 * [64][width] pieces -> [64][count*width], position t*count+j from column t
 * of piece j, optional ReLU. K=N=64 and width%8==0 only; -1 otherwise.
 * Every input is quantized before y is written, so y may overlap the pieces. */
int dpdf_dense_cm_run(dpdf_dense *,const float *const *,int,int,float *,int);
size_t dpdf_dense_bytes(const dpdf_dense *);
dpdf_convop *dpdf_convop_create(const float *,const float *,int,int,int,int,int,int,int,int,int,int,int,int,int);
void dpdf_convop_destroy(dpdf_convop *);
void dpdf_convop_run(dpdf_convop *,const float *,float *);
size_t dpdf_convop_bytes(const dpdf_convop *);
#endif
