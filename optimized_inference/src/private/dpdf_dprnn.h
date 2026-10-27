#ifndef DPDF_DPRNN_H
#define DPDF_DPRNN_H
#include <stddef.h>
#define DPDF_API /* Private implementation declarations. */
#ifdef __cplusplus
extern "C" {
#endif
/* Fixed 64-channel W7A8 DPRNN blocks with FP32 recurrent state. */
enum { DPDF_H=64, DPDF_WEIGHT_FLOATS=87552 };
typedef struct dpdf_block dpdf_block;
int dpdf_has_avx2(void);
size_t dpdf_block_bytes(const dpdf_block *block);
dpdf_block *dpdf_create_int8(int freq,const float *weights,size_t count,
                             float intra_epsilon,float inter_epsilon);
void dpdf_destroy(dpdf_block *block);
/* Channel-major input/output unless the corresponding layout flag is set. */
enum { DPDF_INPUT_FREQ_MAJOR=1, DPDF_OUTPUT_FREQ_MAJOR=2 };
int dpdf_process_layout(const dpdf_block *,const float *,const float *,float *,float *,unsigned);
#ifdef __cplusplus
}
#endif
#endif
