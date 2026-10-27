#ifndef DPDF_FULL_MODEL_H
#define DPDF_FULL_MODEL_H
#include "dpdf_dprnn.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Fixed W7A8 spectral model with FP32 state and selected FP32 operators.
 * Caller verifies the exported weight SHA-256 before calling create.
 * Spectrum: 962 interleaved real/imag float32 values per 480-sample hop.
 * Caller owns stream state and may update input/state in place.
 * No allocation during process.
 */
typedef struct dpdf_model dpdf_model;
DPDF_API const char *dpdf_model_weights_sha256(void);
DPDF_API size_t dpdf_model_weight_count(void);
DPDF_API size_t dpdf_model_arena_bytes(void);
DPDF_API size_t dpdf_model_state_size(void);
/* Initialize/reset exactly dpdf_model_state_size() floats. */
DPDF_API int dpdf_model_init_state(float *state);
DPDF_API dpdf_model *dpdf_model_create(const float *weights, size_t count);
DPDF_API size_t dpdf_model_owned_bytes(const dpdf_model *);
DPDF_API void dpdf_model_destroy(dpdf_model *model);
DPDF_API int dpdf_model_process(dpdf_model *model, const float *spec, const float *state_in,
                                 float *spec_out, float *state_out);
#ifdef __cplusplus
}
#endif
#endif
