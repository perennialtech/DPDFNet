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
size_t dpdf_dense_bytes(const dpdf_dense *);
dpdf_convop *dpdf_convop_create(const float *,const float *,int,int,int,int,int,int,int,int,int,int,int,int,int);
void dpdf_convop_destroy(dpdf_convop *);
void dpdf_convop_run(dpdf_convop *,const float *,float *);
size_t dpdf_convop_bytes(const dpdf_convop *);
#endif
