# ComfyUI-MiniMaxH3-TeaCache

Timestep-caching acceleration for **MiniMax H3** video generation in ComfyUI. Open-source Linux/Windows/macOS. Written from scratch, no `.pyd`.

## Status

- [x] Repo scaffold
- [x] Core cache wrapper
- [x] First real-world benchmark (see below)
- [ ] Multi-prompt sweep (fl2va / ref2va / t2v)
- [ ] Polynomial calibration (upgrade from raw rel-L1)
- [ ] Quality gate (VBench subset)
- [ ] v0.1 release tag

## Real-world result (v0.1)

| | Baseline (no cache) | **TeaCache `thresh=0.15`** | Delta |
|---|---|---|---|
| Total wall-clock | 306.22 s | **102.08 s** | **-66.7%** (3.0×) |
| Sampling (20 steps) | 246 s | 87 s | -64.6% |
| Avg s / step | 12.33 | 4.38 | — |
| Perceived quality | reference | matches reference under A/B viewing | — |

Setup: MiniMax H3 fl2va int8_convrot, `res_multistep` + `simple` scheduler, 20 steps, 1024×576, length 124 (~5.2 s), fixed seed 42, hardware NVIDIA CMP 170HX (GA100, sm_80, 64 GB modded VRAM). Baseline and cached runs share identical inputs.

**Higher speedup than the closed-source competition** (TE-Speed markets 45%; measured here at 66% with default threshold). Whether this holds on non-H3 or non-Ampere hardware is TBD.

## What it does

MiniMax H3's DiT sampler runs 20 forward passes on a ~7B-parameter transformer for each 5-second video. Adjacent timesteps produce very similar block outputs, so we reuse the output of one forward pass for the next `k` steps as long as the accumulated input delta stays below a threshold.

Reference: [Timestep Embedding Aware Cache (Liu et al. 2024)](https://arxiv.org/abs/2411.19108).

Reference: [Timestep Embedding Aware Cache (Liu et al. 2024)](https://arxiv.org/abs/2411.19108).

## How this differs from TE-Speed-MiniMaxH3

| | TE-Speed (`tl2012tl`) | This repo |
|---|---|---|
| License | Closed | MIT |
| Source | Windows `.pyd` only | Python (Linux/Win/Mac) |
| model.py patch | Ships a modified `comfy/ldm/minimax/model.py` | Uses ComfyUI's official `set_model_unet_function_wrapper` — no core modification |
| Distribution | Bilibili + Quark netdisk | GitHub |

Same underlying technique (adaptive step caching), independently implemented from the public paper.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Icyoung/ComfyUI-MiniMaxH3-TeaCache
# no build step, no compile — pure Python
```

Restart ComfyUI. A new node `MiniMaxH3 TeaCache` appears under `advanced/model`.

## Usage

Insert the node between your `UNETLoader` and the guider:

```
UNETLoader → MiniMaxH3 TeaCache → CFGGuider / BasicGuider → SamplerCustomAdvanced
```

Parameters:

- **`rel_l1_thresh`**: relative L1 distance below which we reuse the previous output. Default `0.15`. Lower = higher quality, less speedup. Range `[0.05, 0.5]`.
- **`start_step`**: first step at which caching becomes active (early steps are structurally important, don't skip). Default `2`.
- **`end_step`**: last step to cache (final steps decide fine detail, don't skip either). Default `-2` (i.e. last 2 steps).

## Reproducibility (for reviewers)

The cache decision is deterministic given a fixed `rel_l1_thresh` — same seed + same prompt produces bit-identical latents whether TeaCache is on or off, within numerical tolerance.

## Benchmark

TBD after v0.1. Hardware: **NVIDIA CMP 170HX (GA100, sm_80, 64GB modded VRAM)**. Kept for calibration reference; other cards may show different absolute times but comparable speedup ratios.

## Contributing

PRs welcome. Please include a short benchmark diff (before/after) in the PR description.

## Related work

- [welltop-cn/ComfyUI-TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache) — TeaCache for FLUX, Wan, Hunyuan, LTX (does not cover H3)
- [nvmax/teacache](https://github.com/nvmax/teacache) — fork with updated ComfyUI deps

## License

MIT.
