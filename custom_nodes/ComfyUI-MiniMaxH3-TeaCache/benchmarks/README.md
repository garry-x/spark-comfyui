# Benchmark harness

Two things we measure:

1. **Speed** — wall-clock time per sampling step, and total end-to-end.
2. **Quality** — pixel-space LPIPS and VBench subset scores against a
   TeaCache-off baseline.

## Fixed test setup

* Model: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
* Encoder: `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`
* VAE (video): `minimax_h3_video_vae_fp16.safetensors`
* VAE (audio): `minimax_h3_audio_vae_fp32.safetensors`
* Sampler: `res_multistep`
* Scheduler: `beta`, 20 steps, denoise 1.0
* Prompts: `prompts.jsonl` (10 prompts, mixed styles)
* Seeds: `[0, 1, 2]` (three seeds per prompt)
* Resolution: 1024 × 576, length 121 (5 s @ 24 fps)

## Sweeps

| `rel_l1_thresh` | Expected speedup |
|---|---|
| 0.00 (off) | baseline |
| 0.05 | ~15% |
| 0.10 | ~25% |
| 0.15 | ~40% |
| 0.20 | ~50% |
| 0.25 | ~55% |

## Run

```bash
cd benchmarks
python run.py \
    --comfy-url http://hp-z4-server:8188 \
    --workflow ../workflows/fl2va-bench.json \
    --thresholds 0,0.05,0.10,0.15,0.20,0.25 \
    --output output/
```

Output is a CSV with columns:
`prompt_id, seed, thresh, real_steps, reuse_steps, wall_time_s, lpips_vs_off`.

## Hardware note

The 170HX in the reference machine is a mining card with mod-VRAM (64GB
total). Its FP16 throughput is ~85% of an RTX 3090, so absolute wall-clock
numbers are conservative. Speedup **ratios** transfer well to 4090/5090/H100.
