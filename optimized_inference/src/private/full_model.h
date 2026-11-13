#ifndef DPDF_FULL_MODEL_H
#define DPDF_FULL_MODEL_H
#include "dpdf_dprnn.h"
#include <stdint.h>
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
/* Creation flags. LANES runs the DF branch's sections on a private arena in
 * a two-lane schedule; WORKER (which requires LANES) gives that lane its own
 * thread. LANES alone runs both lanes on the calling thread. */
#define DPDF_MODEL_LANES 1u
#define DPDF_MODEL_WORKER 2u
DPDF_API dpdf_model *dpdf_model_create(const float *weights, size_t count, unsigned flags);
DPDF_API size_t dpdf_model_owned_bytes(const dpdf_model *);
DPDF_API void dpdf_model_destroy(dpdf_model *model);
DPDF_API int dpdf_model_process(dpdf_model *model, const float *spec, const float *state_in,
                                 float *spec_out, float *state_out);
/* Test support: fill the arenas with distinct NaN payloads, so a result that
 * depends on a previous hop's arena contents (or a missing lane copy) shows. */
DPDF_API void dpdf_model_scramble(dpdf_model *model, uint32_t seed);
/* Test support: a worker context runs its next hops sequentially. */
DPDF_API void dpdf_model_pause(dpdf_model *model, int hops);
#ifdef __cplusplus
}
#endif
#endif
