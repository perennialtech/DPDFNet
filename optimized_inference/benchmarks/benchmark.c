#define _POSIX_C_SOURCE 200809L
#include "../tools/support.h"
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
static int64_t monotonic_ns(void) {
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return c.QuadPart / f.QuadPart * INT64_C(1000000000) + c.QuadPart % f.QuadPart * INT64_C(1000000000) / f.QuadPart;
}
static int64_t cpu_ns(void) { /* Kernel plus user time, in 100 ns units. */
    FILETIME created, exited, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) exit(1);
    ULARGE_INTEGER k = {.LowPart = kernel.dwLowDateTime, .HighPart = kernel.dwHighDateTime};
    ULARGE_INTEGER u = {.LowPart = user.dwLowDateTime, .HighPart = user.dwHighDateTime};
    return (int64_t)(k.QuadPart + u.QuadPart) * 100;
}
/* A high-resolution timer: plain waits round up to the 15.6 ms timer tick. */
static void sleep_until(int64_t deadline) {
    static HANDLE timer;
    if (!timer && !(timer = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS))) exit(1);
    for (int64_t left; (left = deadline - monotonic_ns()) > 0;) {
        LARGE_INTEGER due = {.QuadPart = -((left + 99) / 100)};
        if (!SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE)) exit(1);
        WaitForSingleObject(timer, INFINITE);
    }
}
static size_t rss(void) {
    PROCESS_MEMORY_COUNTERS c;
    return GetProcessMemoryInfo(GetCurrentProcess(), &c, sizeof(c)) ? c.WorkingSetSize : 0;
}
#else
#include <time.h>
#include <unistd.h>
static int64_t nanoseconds(clockid_t clock) {
    struct timespec t; if (clock_gettime(clock, &t)) { perror("clock_gettime"); exit(1); }
    return (int64_t)t.tv_sec * INT64_C(1000000000) + t.tv_nsec;
}
static int64_t monotonic_ns(void) { return nanoseconds(CLOCK_MONOTONIC); }
static int64_t cpu_ns(void) { return nanoseconds(CLOCK_PROCESS_CPUTIME_ID); }
static void sleep_until(int64_t deadline) {
    struct timespec t = {deadline / INT64_C(1000000000), deadline % INT64_C(1000000000)};
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL)) {}
}
static size_t rss(void) {
    FILE *f = fopen("/proc/self/statm", "r"); unsigned long total, resident;
    if (!f) return 0;
    int ok = fscanf(f, "%lu %lu", &total, &resident) == 2; fclose(f);
    return ok ? (size_t)resident * (size_t)sysconf(_SC_PAGESIZE) : 0;
}
#endif
static int compare(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y);
}
int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) {
        fprintf(stderr, "Usage: %s 2|8 weights.f32 [hops=1000] [paced|continuous][-single] [spectra.f32]\n", argv[0]); return 2;
    }
    dpdfnet_model_id id = (dpdfnet_model_id)atoi(argv[1]);
    const dpdfnet_model_info *info = dpdfnet_get_model_info(id);
    size_t hops = argc > 3 ? (size_t)strtoul(argv[3], NULL, 10) : 1000;
    const char *mode = argc > 4 ? argv[4] : "paced";
    int paced = !strncmp(mode, "paced", 5), single = strstr(mode, "-single") != NULL;
    const char *base = paced ? "paced" : "continuous";
    if (!info || hops < 100 || hops > 1000000 || strncmp(mode, base, strlen(base)) ||
        strcmp(mode + strlen(base), single ? "-single" : "")) return 2;
    size_t frame_count = 256;
    FILE *input = argc > 5 ? fopen(argv[5], "rb") : NULL;
    if (argc > 5) {
        if (!input || fseek(input, 0, SEEK_END)) return 1;
        long bytes = ftell(input); if (bytes <= 0 || bytes % (962 * 4)) return 1;
        frame_count = (size_t)bytes / (962 * 4); rewind(input);
    }
    float *frames = malloc(frame_count * 962 * sizeof(float));
    double *times = malloc(hops * sizeof(double));
    float *state = calloc(info->state_floats, sizeof(float));
    if (!frames || !times || !state) return 1;
    if (input) {
        if (fread(frames, sizeof(float), frame_count * 962, input) != frame_count * 962) return 1;
        fclose(input);
    } else for (size_t i = 0; i < frame_count; ++i) synthetic_spectrum(frames + i * 962, i, 1.0f);
    for (size_t i = 0; i < frame_count * 962; ++i) if (!isfinite(frames[i])) return 1;
    size_t before = rss();
    float *weights = read_weights(argv[2], info);
    dpdfnet_model *m = weights ? dpdfnet_create_with_flags(id, weights, info->weight_floats,
                                                               single ? DPDFNET_CREATE_SINGLE_THREAD : 0) : NULL;
    free(weights);
    if (!m || dpdfnet_init_state(id, state)) return 1;
    double sum = 0; size_t over = 0, late = 0;
    int64_t cpu_start = 0; /* Process CPU over the timed hops, helper spins included. */
    float output[962];
    int64_t deadline = monotonic_ns();
    for (size_t i = 0; i < hops + 120; ++i) {
        if (i == 120) cpu_start = cpu_ns();
        if (paced) sleep_until(deadline);
        int64_t t0 = monotonic_ns();
        if (dpdfnet_process(m, frames + (i % frame_count) * 962, state, output, state)) return 1;
        int64_t t1 = monotonic_ns();
        if (i >= 120) {
            double ms = (double)(t1 - t0) / 1e6; times[i - 120] = ms; sum += ms;
            over += ms > 10;
            late += paced && t1 > deadline + INT64_C(10000000);
        }
        deadline += INT64_C(10000000);
        if (paced && t1 > deadline) deadline = t1; /* Record miss before restarting cadence. */
    }
    double cpu = (double)(cpu_ns() - cpu_start) / 1e6;
    size_t after = rss(); qsort(times, hops, sizeof(double), compare);
    printf("{\"model\":\"%s\",\"input\":\"%s\",\"mode\":\"%s\",\"single_thread\":%s,\"hops\":%zu,"
           "\"mean_ms\":%.6f,\"cpu_ms\":%.6f,\"p99_ms\":%.6f,\"max_ms\":%.6f,"
           "\"over_10ms\":%zu,\"late\":%zu,\"owned_bytes\":%zu,\"rss_bytes\":%zu,\"rss_increment_bytes\":%zd}\n",
           info->name, input ? "file" : "synthetic", base, single ? "true" : "false", hops,
           sum / hops, cpu / hops, times[(hops * 99) / 100], times[hops - 1], over, late,
           dpdfnet_owned_bytes(m), after, (ptrdiff_t)after - (ptrdiff_t)before);
    dpdfnet_destroy(m); free(state); free(times); free(frames);
    return 0;
}
