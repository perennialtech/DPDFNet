# Architecture and maintenance

The public runtime selects one of two fixed graph backends. Each backend is
compiled from shared kernels, a model-specific graph, and its final assembly
specializations. Backend symbols are isolated at compile time; neither graph
can accidentally bind to the other model's kernels.

## Numerical implementation

Quantized matrices use signed **7-bit weights** and dynamically scaled **8-bit
activations**, with integer dot products and FP32 scale/bias application.
The weight restriction prevents saturation in the AVX2 packed multiply path.
DPRNN, dense/grouped projections and eligible 1x1 convolutions use this path.
Other convolutions, normalization, complex filtering and recurrent state
remain FP32. DPRNN gates use the validated fitted degree-5 approximation.

Adjacent DPRNN blocks retain frequency-major features, avoiding repeated
transposes. Matrix tiles use packed 32-byte operands; kernels specialize the
fixed graph's common 64-input projections. Model 2 uses fused four-row
paired projections on both AVX2 and AVX-VNNI; model 8 uses them on AVX-VNNI
only (its original AVX2 tiled reduction benchmarks faster). Model 2 also
uses a 256-input assembly projection and AVX2 arithmetic surrounding unchanged
scalar libm calls in its generated GRU cells. Model 8 retains its accepted
fixed-row kernels and compiler profile.

Model-specific choices in shared C sources are selected by the private
`DPDF_MODEL_SIZE` compile definition. These branches preserve the accepted
operation order of each graph. Do not replace explicit operations with FMA
or change quantization rounding without new numerical and audio evidence.

## Source boundaries

- `runtime.c`, `backend.h`, and `model_backend.c` implement the public API and
  select the fixed W7A8 configuration.
- `common/int8.c`, `avx2.c`, and assembly perform quantization, projection,
  activation and normalization operations.
- `common/dpdf_dprnn.c` manages the fixed DPRNN blocks; `extended_ops.c` manages
  dense/grouped/convolution storage and graph helpers.
- `src/models/*/generated_model.c` contains explicit fixed graph operations,
  weight offsets, state layout, and normalization seeds. These files belong to
  the supplied model/weight hashes; they are not a general ONNX interpreter.
- `private/` holds kernel declarations and namespace mappings. These headers
  are not installed and do not define a consumer ABI.

Independent scalar reference calculations remain in contract tests to check
quantized arithmetic, convolutions and normalization. The production graph
is fixed to W7A8; the only CPU dispatch selects AVX-VNNI or AVX2 kernels.

## Validation expectations

Run release and sanitizer contracts after kernel changes. Preserve guarded
assembly tests: sanitizers cannot instrument the `.S` files. Use the saved
stream hashes to compare every spectrum, full state history and synthesized
PCM across the 65 fixtures, rather than checking only the last frame.

The fixed graphs are paired with `models/*/manifest.json`. A changed model or
weight layout requires new graph generation, manifests and reference outputs.
Source reorganization, symbol changes and new compiler flags require fresh
model-8 PGO training and a latency check on the target machine.

Keep benchmark jobs isolated from builds and correctness tests. Use balanced
process order, equal input/warmup/cadence, and same-session comparisons for
speedup claims. Per-hop samples are not independent experiments. An observed
maximum is a finite measurement, not a worst-case execution-time bound.
