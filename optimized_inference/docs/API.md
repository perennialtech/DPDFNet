# C API and audio integration

[`include/dpdfnet.h`](../include/dpdfnet.h) is the complete public API, ABI
version 1. The same library supports model IDs `DPDFNET_MODEL_2` and
`DPDFNET_MODEL_8`. There is one optimized precision configuration.

## Context lifecycle

1. Query `dpdfnet_get_model_info(id)` and check `dpdfnet_cpu_supported()`.
2. Load the corresponding little-endian IEEE-754 float32 weights. Check SHA-256
   against `weights_sha256` or the bundled manifest before using them.
3. Call `dpdfnet_create(id, weights, info->weight_floats)`. It copies/repacks the
   required weights; the source buffer may be freed after successful creation.
   The context starts one helper thread; see [worker thread](#worker-thread).
4. Allocate `info->state_floats` floats and call `dpdfnet_init_state(id, state)`.
   This initializes normalization seeds as well as zeroed recurrent entries.
5. Call `dpdfnet_process` once per hop, passing the state from the preceding hop.
6. Reset between unrelated streams. Free the state and destroy the context when
   finished. `dpdfnet_destroy(NULL)` is safe.

Unknown model IDs return NULL metadata. Creation returns NULL for invalid
arguments/counts, nonfinite weights, unsupported CPUs or allocation failure.
State initialization and processing return 0 on success and -1 for invalid
IDs/NULL pointers as described in the header. The API cannot infer buffer
lengths from pointers; callers must provide the documented sizes.

| Model | Weight floats | State floats | State bytes |
| --- | ---: | ---: | ---: |
| `dpdfnet2_48khz_hr` | 2,582,444 | 56,436 | 225,744 |
| `dpdfnet8_48khz_hr` | 3,633,068 | 90,228 | 360,912 |

## Processing contract

The input/output each contain **962 float32 values**, shape `[481,2]`, with
real/imaginary pairs. This is the positive-frequency half of an unnormalized
960-point real FFT. One hop advances 480 samples at 48,000 Hz.

Spectrum and state may each be updated in place. Other buffer overlaps are
invalid. All inputs/state/weights must be finite; processing does not scan the
entire state for NaNs. Four-byte float alignment is sufficient: 32-byte caller
alignment is not required. Keep the process floating-point control settings
stable; the validated release uses round-to-nearest and the host's default
denormal handling.

A context owns mutable scratch memory. Do not call the same context
simultaneously or destroy it during processing. Separate contexts can run
concurrently and can select different models. State belongs to the caller;
there is no hidden stream state.

`dpdfnet_process` performs no allocation. Weight loading, context creation,
reset, and memory allocation should happen outside the audio callback. Some
kernel scratch uses the caller's stack; reserve at least 1 MiB of available
stack when creating an inference thread. Preallocate spectral/audio buffers
and avoid worker pool contention in the callback.

## Worker thread

A context owns one helper thread. Each hop runs the model's ERB and
deep-filter branches at the same time, one on the calling thread and one on
the helper. Spectrum and state outputs are bit-identical to a single-threaded
context, in every rounding and denormal mode.

`dpdfnet_create_with_flags(id, weights, count, DPDFNET_CREATE_SINGLE_THREAD)`
opts out: that context runs every hop on the calling thread, with no helper,
less CPU time and less memory. Flags of 0 behave like `dpdfnet_create`, and
unknown flags return NULL. A context is also single-threaded when the
creating thread may use only one CPU, or when the helper thread cannot be
started; creation does not fail for either reason.

The calling contract is unchanged: one `dpdfnet_process` call at a time per
context, from any thread. The helper only runs work that call hands it, and
the call returns after both branches finish.

- **Cost.** On two free physical cores a hop took 24–45% less wall time in
  our tests, for 10–65% more total CPU time; see [performance](PERFORMANCE.md). Model 8
  owns 977,600 more heap bytes than a single-threaded context, model 2
  879,296.
- **Cores.** The helper inherits the creating thread's CPU affinity, scheduling
  policy and priority when the context is created. Create the context from the
  thread that will process audio, or from one with the same settings: an
  ordinary-priority helper serving a real-time audio thread can be preempted.
  When that is impractical (an audio callback thread owned by a framework),
  find the `dpdfnet-lane` thread under `/proc/self/task` and set its
  priority and affinity to match, or use `DPDFNET_CREATE_SINGLE_THREAD`.
  The thread is named `dpdfnet-lane`. Two SMT siblings of one core gave only
  about 12% in our tests, for nearly double the CPU time. With a single allowed
  CPU, hops take as long as without the helper.
- **Idle behaviour.** The helper sleeps in the kernel between hops. When hops
  arrive at a steady cadence (100 µs to 100 ms, low jitter), it wakes shortly
  before the next expected hop and spins until it starts, so its core is
  already awake. That spin is most of the extra CPU time, and lasts at most
  about 0.8 ms per hop. Within a hop the
  two threads spin while waiting for each other, and sleep only after about
  1 ms. Spinning threads yield every 20 µs or so, so a spinning thread never
  holds a shared CPU its partner needs.
- **Worst case.** If the helper has not started its branch by the time the
  caller finishes the other one, the caller runs it too, so a late helper costs
  about one single-threaded hop. A helper preempted partway through its branch
  delays the hop until it is rescheduled, so under CPU contention an
  individual hop can still exceed single-threaded time.
- **Back-off.** When such delays recur, or the helper keeps arriving too late
  to take any work, the context runs hops on the calling thread alone for a
  while (64 hops, doubling to 4,096 while the problem persists) and then
  tries the helper again. Outputs are identical either way. On a quiet
  machine this does not trigger.
- **Floating point.** The helper runs with the caller's MXCSR rounding and
  denormal controls, and exception flags it raises are merged into the
  caller's MXCSR.
- **Signals and faults.** The helper blocks all signals, so they keep going to
  the application's threads. A fault on the helper (such as an unmasked
  floating-point exception, since the helper adopts the caller's MXCSR
  masks) therefore ends the process without reaching the application's
  handlers. Keep floating-point exceptions masked, as the release is validated.
- **Fork.** A child process has no helper. A context inherited across `fork`
  runs single-threaded there, with identical results, and can be destroyed
  normally; the helper's stack is not reclaimed in the child.

`dpdfnet_destroy` stops and joins the helper.

## FFT, synthesis and delay

Use a 960-sample analysis frame, a 480-sample hop and the float32 window:

```text
w[n] = sin((pi/2) * sin(pi*(n+0.5)/960)^2),  n = 0..959
```

Initialize analysis history with 480 zeros. Frame zero contains that history
and the first 480 input samples. Advance one hop, take a real FFT, and pass
interleaved real/imaginary bins to the model. Keep DC and Nyquist imaginary
components zero for a real input. Do not normalize the forward FFT.

For synthesis, use a 960-point inverse real FFT with `1/960` normalization,
multiply by the same window, overlap-add, and emit 480 samples per hop.
The complete pipeline has 2,400 samples (**50 ms**) of algorithmic delay.
That signal delay is distinct from CPU execution time.

For offline files, feed six extra zero hops to drain the pipeline, discard the
first 2,400 output samples and keep the original input length. The Python
[WAV example](../examples/enhance_wav.py) implements this exact convention and
rejects inputs other than 48 kHz mono. Live applications keep the causal delay
and drain only when ending a stream.

The Python [ctypes adapter](../examples/dpdfnet.py) verifies weight checksums,
owns a state vector and returns a borrowed output array overwritten by the
next call. Copy returned output when retaining it. Use its context manager or
call `close()` explicitly.

## Memory accounting

`dpdfnet_owned_bytes` reports context-owned heap allocations, including the
16-byte public wrapper on the supported 64-bit ABI. It excludes caller state,
spectra, thread stack, code, allocator overhead/retention, and temporary source
weights during creation. It is neither model file size nor process RSS.

Context creation temporarily needs the original float32 weights alongside
packed storage. Budget that startup peak separately from steady-state memory.
