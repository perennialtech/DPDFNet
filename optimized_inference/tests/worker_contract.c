/* Worker-thread contexts through the public API: bit-identical spectra,
 * state and raised exception flags against single-threaded contexts in every
 * rounding mode with and without FTZ/DAZ, in-place and out-of-place buffers,
 * and repeated creation and destruction. */
#include "../tools/support.h"
#include <fenv.h>
#include <immintrin.h>
#include <string.h>
#define CHECK(expr) do { if (!(expr)) { fprintf(stderr, "Failed line %d: %s\n", __LINE__, #expr); exit(1); } } while (0)
static const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
int main(int argc, char **argv) {
    CHECK(argc == 3);
    if (!dpdfnet_cpu_supported()) { puts("AVX2/FMA unavailable; skipped"); return 0; }
    const unsigned original = _mm_getcsr();
    for (int index = 0; index < 2; ++index) {
        dpdfnet_model_id id = index ? DPDFNET_MODEL_8 : DPDFNET_MODEL_2;
        const dpdfnet_model_info *info = dpdfnet_get_model_info(id);
        float *w = read_weights(argv[index + 1], info);
        CHECK(w);
        CHECK(!dpdfnet_create_with_flags(id, w, info->weight_floats, 2u));
        CHECK(!dpdfnet_create_with_flags(id, w, info->weight_floats - 1, DPDFNET_CREATE_WORKER_THREAD));
        CHECK(!dpdfnet_create_with_flags((dpdfnet_model_id)3, w, info->weight_floats, DPDFNET_CREATE_WORKER_THREAD));
        /* Contexts that never process, or process once, still shut down. */
        for (int round = 0; round < 24; ++round) {
            dpdfnet_model *m = dpdfnet_create_with_flags(id, w, info->weight_floats, DPDFNET_CREATE_WORKER_THREAD);
            CHECK(m);
            if (round % 2) {
                float *s = malloc(info->state_floats * sizeof(float)), spec[962];
                CHECK(s && dpdfnet_init_state(id, s) == 0);
                synthetic_spectrum(spec, (size_t)round, 1.0f);
                CHECK(dpdfnet_process(m, spec, s, spec, s) == 0);
                free(s);
            }
            dpdfnet_destroy(m);
        }
        dpdfnet_model *single = dpdfnet_create_with_flags(id, w, info->weight_floats, 0);
        dpdfnet_model *worker = dpdfnet_create_with_flags(id, w, info->weight_floats, DPDFNET_CREATE_WORKER_THREAD);
        CHECK(single && worker);
        CHECK(dpdfnet_owned_bytes(worker) > dpdfnet_owned_bytes(single));
        free(w);
        size_t n = info->state_floats;
        float *a = malloc(n * sizeof(float)), *b = malloc(n * sizeof(float)), *next = malloc(n * sizeof(float));
        CHECK(a && b && next);
        float x[962], y[962], z[962];
        const float scales[] = {1.0f, 0.001f, 8.0f, 0.0f};
        for (int mode = 0; mode < 8; ++mode) {
            _mm_setcsr((original & ~0x8040u) | ((mode & 4) ? 0x8040u : 0u));
            CHECK(fesetround(modes[mode & 3]) == 0);
            const unsigned controls = _mm_getcsr() & 0xffc0u;
            CHECK(dpdfnet_init_state(id, a) == 0 && dpdfnet_init_state(id, b) == 0);
            for (size_t frame = 0; frame < 40; ++frame) {
                synthetic_spectrum(x, frame + 97 * (size_t)mode, scales[(frame / 10) % 4]);
                _mm_setcsr(_mm_getcsr() & ~0x3fu);
                CHECK(dpdfnet_process(single, x, a, y, a) == 0);
                const unsigned single_flags = _mm_getcsr() & 0x3fu;
                _mm_setcsr(_mm_getcsr() & ~0x3fu);
                if (frame % 2) {
                    memcpy(z, x, sizeof(x));
                    CHECK(dpdfnet_process(worker, z, b, z, b) == 0);
                } else {
                    CHECK(dpdfnet_process(worker, x, b, z, next) == 0);
                    memcpy(b, next, n * sizeof(float));
                }
                const unsigned worker_flags = _mm_getcsr() & 0x3fu;
                CHECK(!memcmp(y, z, sizeof(y)) && !memcmp(a, b, n * sizeof(float)));
                CHECK(worker_flags == single_flags);
            }
            CHECK((_mm_getcsr() & 0xffc0u) == controls);
        }
        _mm_setcsr(original);
        dpdfnet_destroy(single); dpdfnet_destroy(worker);
        free(a); free(b); free(next);
    }
    puts("Worker contexts match single-threaded contexts in every FP control mode");
    return 0;
}
