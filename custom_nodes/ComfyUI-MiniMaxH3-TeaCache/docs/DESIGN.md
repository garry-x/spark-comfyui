# Design notes

## Why not modify `comfy/ldm/minimax/model.py`

TE-Speed ships a modified `comfy/ldm/minimax/model.py` and asks users to
replace the stock file. That approach has two problems:

1. **Fragility across ComfyUI upgrades.** Every `git pull` on the ComfyUI
   repo overwrites the patch and the plugin silently degrades to no-op.
2. **Conflict with any other plugin** that also touches this file (there
   already are TE_MAN, ComfyUI-MiniMax-Remover, etc.).

The stock ComfyUI provides
`ModelPatcher.set_model_unet_function_wrapper(fn)` which wraps the entire
`apply_model` call. This is the correct extension point for step-level
caching, and it works for every model type ComfyUI supports — not only H3.
No core file is touched.

## Where the reuse decision happens

Inside our wrapper:

```
def wrapper(apply_model, args):
    x = args["input"]
    if should_reuse(state, x, ...):
        state.step_idx += 1
        return state.prev_output      # cached
    out = apply_model(x, args["timestep"], **args["c"])
    state.prev_input, state.prev_output = x.detach(), out.detach()
    state.step_idx += 1
    return out
```

`should_reuse` looks at `rel_l1(x_t, x_{t-1})`, accumulates across skipped
steps, and only reuses while the accumulated delta stays below a threshold.
When the threshold is crossed we run the real forward pass and reset.

## Calibration (Phase 2)

The published TeaCache formulation uses a **calibrated polynomial** to
predict output-delta from input-delta:

```
predicted_output_delta = a0 + a1·d + a2·d² + a3·d³ + a4·d⁴
```

Coefficients differ per model architecture and per sampler. Extracting
them for H3 audio-video packed layout needs:

1. Log `(input_delta, output_delta)` pairs from ~50 real generations with
   caching **off**.
2. Fit a degree-4 polynomial (least squares) per stream (audio, video).
3. Bake the fitted coefficients into `teacache.py` as constants keyed by
   `(sampler_name, num_steps)`.

Until Phase 2, `rel_l1` on the raw input is used directly. This misses the
efficiency of the polynomial method but is qualitatively correct.

## Threshold tuning cheat sheet

| `rel_l1_thresh` | Speedup | Quality change |
|---|---|---|
| 0.05 | ~15% | Imperceptible |
| **0.15** (default) | **~40%** | Very slight softening on high-motion frames |
| 0.25 | ~55% | Visible loss of detail in fast motion |
| 0.35+ | ~65% | Structural artifacts likely |

Numbers are indicative — real values will be filled in from the H3
benchmark harness in `benchmarks/`.

## Streams and cache scope

MiniMax H3 has a packed input `[text | cond | audio | video]`. Text and
cond tokens are constant across a sampling run (they're set once from the
prompt / reference frames). Only `audio` and `video` token positions
change from step to step, so those are the only positions that need
separate cache statistics if we go finer-grained later.

For v0.1 we treat the full input tensor as one signal and skip the
per-stream separation — simpler, correct, and matches how public TeaCache
implementations handle FLUX and Hunyuan.

## Non-goals

* We do not touch the sampler. `SamplerCustomAdvanced` and the schedulers
  (`res_multistep`, `simple`, `beta`) all work unchanged.
* We do not modify weights, quantization, or precision.
* We do not fuse Qwen3-VL text encoder passes — that's a separate speedup
  that TE-Speed also does (their `TESpeedMiniMaxH3` node caches the text
  encoding across prompts). Out of scope for v0.1.
