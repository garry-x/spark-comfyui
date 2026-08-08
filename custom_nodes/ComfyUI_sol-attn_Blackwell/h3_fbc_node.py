"""FirstBlockCache for MiniMax H3 in ComfyUI — single-GPU port of the Sol Engine cache line.

Algorithm (NVlabs/Sana sol-engine branch, models/minimax_h3/optimized/cache_line.py):
run block 0 every step and compare its residual to the previous step's:
    diff = mean|res_t - res_{t-1}| / mean|res_{t-1}|
If diff <= threshold, the remaining blocks are skipped and the previous step's
tail residual (h_final - h_after_block0) is reapplied instead. Sol Engine measured
~69% of block-stack calls deleted at threshold 0.08 (50 steps, 2.58x hot-path).

Implemented via ComfyUI's patches_replace["dit"] per-block hook — no core edits:
  block 0    : compute residual, make the skip decision for this step
  blocks 1..N-2 : identity when skipping
  block N-1  : apply cached tail residual when skipping, else record it
"""

import logging
import os
import torch

logger = logging.getLogger(__name__)
_PROBE = os.environ.get("H3_FBC_PROBE", "") == "1"


class H3FirstBlockCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "threshold": ("FLOAT", {
                    "default": 0.08, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Skip remaining blocks when the first-block residual changed less "
                               "than this fraction vs the previous step. Sol Engine reference: 0.08 "
                               "at 50 steps. Higher = faster, more quality risk.",
                }),
                "start_step": ("INT", {
                    "default": 2, "min": 0, "max": 100,
                    "tooltip": "Always run all blocks for this many initial steps.",
                }),
                "end_dense_steps": ("INT", {
                    "default": 2, "min": 0, "max": 100,
                    "tooltip": "Always run all blocks for this many final steps (detail refinement).",
                }),
                "max_consecutive_skips": ("INT", {
                    "default": 2, "min": 1, "max": 20,
                    "tooltip": "Force a full compute after this many skipped steps in a row — "
                               "bounds tail-residual drift on short (20-step) schedules.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/optimization"
    DESCRIPTION = ("FirstBlockCache for MiniMax H3 (Sol Engine cache line, single-GPU port). "
                   "Skips most DiT blocks on steps where the latent is changing slowly.")

    def patch(self, model, threshold, start_step, end_dense_steps, max_consecutive_skips):
        m = model.clone()
        n_blocks = len(m.get_model_object("diffusion_model").blocks)

        state = {
            "last_sigma": None, "step": -1, "total_steps": None,
            "prev_first_residual": None, "h_after_first": None,
            "tail_residual": None, "skip": False, "consec": 0,
            "computed": 0, "skipped": 0,
        }

        def _sync_step(topts):
            """Track step transitions / new runs from the current sigma."""
            sig = topts.get("sigmas", None)
            cur = float(sig.max().item()) if isinstance(sig, torch.Tensor) else None
            if cur is None:
                return
            sched = topts.get("sample_sigmas", None)
            if isinstance(sched, torch.Tensor):
                state["total_steps"] = sched.shape[-1] - 1
            last = state["last_sigma"]
            if last is None or cur > last + 1e-9:  # sigmas decrease within a run
                state.update(step=0, prev_first_residual=None, h_after_first=None,
                             tail_residual=None, skip=False, consec=0, computed=0, skipped=0)
            elif cur < last - 1e-9:
                state["step"] += 1
            state["last_sigma"] = cur

        def first_block(args, extra):
            _sync_step(args["transformer_options"])
            # H3 blocks accumulate into the stream in place (addcmul_) and return the same
            # tensor object — capture a copy of the input or the residual is identically zero.
            h_before = args["img"].clone()
            h_out = extra["original_block"](args)["img"]
            residual = h_out - h_before
            if _PROBE and state["step"] <= 1:
                r = residual.abs().float()
                logger.info(f"[H3-FBC-PROBE] step {state['step']} block 0: "
                            f"|res|sum={r.sum().item():.3e} max={r.max().item():.3e}")

            prev = state["prev_first_residual"]
            skip = False
            in_tail = (state["total_steps"] is not None
                       and state["step"] >= state["total_steps"] - end_dense_steps)
            if (prev is not None and state["tail_residual"] is not None
                    and state["step"] >= start_step and not in_tail
                    and state["consec"] < max_consecutive_skips):
                d = (residual - prev).abs().float()
                p = prev.abs().float()
                ratio_global = (d.sum() / p.sum().clamp(min=1e-8)).item()
                # H3 packs text/audio prefix rows with outlier magnitudes into the same
                # sequence; a global mean ratio is dominated by them. Use a per-row
                # normalized ratio over the trailing 60% of rows (the video segment).
                tail = slice(int(d.shape[0] * 0.4), None)
                row_ratio = (d[tail].mean(dim=-1) /
                             p[tail].mean(dim=-1).clamp(min=1e-8))
                ratio = row_ratio.mean().item()
                skip = threshold > 0 and ratio <= threshold
                logger.info(f"[H3-FBC] step {state['step']}: row_ratio={ratio:.4f} "
                            f"(p50={row_ratio.median().item():.4f} "
                            f"p90={row_ratio.quantile(0.9).item():.4f} "
                            f"global={ratio_global:.6f}) -> {'skip' if skip else 'compute'}")

            state["prev_first_residual"] = residual
            state["skip"] = skip
            state["consec"] = state["consec"] + 1 if skip else 0
            # clone: later blocks mutate h_out in place, and the tail residual needs
            # the true post-block-0 state
            state["h_after_first"] = None if skip else h_out.clone()
            state["skipped" if skip else "computed"] += 1
            return {"img": h_out}

        def make_mid_block(idx):
            def mid(args, extra):
                if state["skip"]:
                    return {"img": args["img"]}
                pre = args["img"].clone() if (_PROBE and state["step"] <= 1 and idx <= 6) else None
                out = extra["original_block"](args)
                if pre is not None:
                    r = (out["img"] - pre).abs().float()
                    logger.info(f"[H3-FBC-PROBE] step {state['step']} block {idx}: "
                                f"|res|sum={r.sum().item():.3e} max={r.max().item():.3e}")
                return out
            return mid

        def last_block(args, extra):
            if state["skip"]:
                return {"img": args["img"] + state["tail_residual"]}
            out = extra["original_block"](args)
            state["tail_residual"] = out["img"] - state["h_after_first"]
            state["h_after_first"] = None
            if state["step"] > 0 and (state["computed"] + state["skipped"]) % 10 == 0:
                logger.info(f"[H3-FBC] steps computed={state['computed']} skipped={state['skipped']}")
            return out

        m.set_model_patch_replace(first_block, "dit", "double_block", 0)
        for i in range(1, n_blocks - 1):
            m.set_model_patch_replace(make_mid_block(i), "dit", "double_block", i)
        m.set_model_patch_replace(last_block, "dit", "double_block", n_blocks - 1)
        logger.info(f"[H3-FBC] installed on {n_blocks} blocks | threshold={threshold} start_step={start_step}")
        return (m,)


NODE_CLASS_MAPPINGS = {"H3FirstBlockCache": H3FirstBlockCache}
NODE_DISPLAY_NAME_MAPPINGS = {"H3FirstBlockCache": "MiniMax H3 FirstBlockCache (Sol Engine)"}
