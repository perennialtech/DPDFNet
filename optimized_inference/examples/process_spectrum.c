/* Raw spectrum example; FFT, windowing and PCM synthesis belong to the host. */
#include "../tools/support.h"
#ifdef _WIN32
#include <windows.h>
/* Windows stat() reports no inode, so compare volume and file IDs. */
static int file_id(const char *path, BY_HANDLE_FILE_INFORMATION *info) {
    HANDLE h = CreateFileA(path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           NULL, OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE) return 0;
    int ok = GetFileInformationByHandle(h, info);
    CloseHandle(h);
    return ok;
}
static int same_file(const char *a, const char *b) {
    BY_HANDLE_FILE_INFORMATION x, y;
    return file_id(a, &x) && file_id(b, &y) && x.dwVolumeSerialNumber == y.dwVolumeSerialNumber &&
           x.nFileIndexHigh == y.nFileIndexHigh && x.nFileIndexLow == y.nFileIndexLow;
}
#else
#include <sys/stat.h>
static int same_file(const char *a, const char *b) {
    struct stat x, y;
    return !stat(a, &x) && !stat(b, &y) && x.st_dev == y.st_dev && x.st_ino == y.st_ino;
}
#endif
int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s 2|8 weights.f32 input_spectra.f32 output_spectra.f32\n", argv[0]);
        return 2;
    }
    dpdfnet_model_id id = (dpdfnet_model_id)atoi(argv[1]);
    const dpdfnet_model_info *info = dpdfnet_get_model_info(id);
    if (!info) { fputs("Model ID must be 2 or 8.\n", stderr); return 2; }
    if (same_file(argv[3], argv[4])) {
        fputs("Input and output must be different files.\n", stderr); return 2;
    }
    float *weights = read_weights(argv[2], info);
    dpdfnet_model *m = weights ? dpdfnet_create(id, weights, info->weight_floats) : NULL;
    free(weights);
    float *state = calloc(info->state_floats, sizeof(float));
    FILE *input = NULL, *output = NULL;
    int status = 1;
    if (!m || !state || dpdfnet_init_state(id, state)) goto cleanup;
    input = fopen(argv[3], "rb");
    if (!input) { perror(argv[3]); goto cleanup; }
    output = fopen(argv[4], "wb");
    if (!output) { perror(argv[4]); goto cleanup; }
    float spectrum[962];
    size_t count, hops = 0;
    while ((count = fread(spectrum, 1, sizeof(spectrum), input)) == sizeof(spectrum)) {
        for (size_t i = 0; i < 962; ++i) if (!isfinite(spectrum[i])) goto cleanup;
        if (dpdfnet_process(m, spectrum, state, spectrum, state) ||
            fwrite(spectrum, sizeof(float), 962, output) != 962) goto cleanup;
        ++hops;
    }
    if (count || ferror(input)) { fputs("Truncated spectrum or input error.\n", stderr); goto cleanup; }
    printf("Processed %zu hops with %s.\n", hops, info->name);
    status = 0;
cleanup:
    if (input && fclose(input)) status = 1;
    if (output && fclose(output)) status = 1;
    free(state);
    dpdfnet_destroy(m);
    return status;
}
