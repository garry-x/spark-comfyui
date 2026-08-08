"""Core TeaCache decision logic.

TeaCache (Liu et al., 2024, arxiv:2411.19108) accelerates diffusion sampling
by skipping redundant model forward passes: adjacent timesteps produce very
similar outputs, so we can reuse a previous output plus a small delta.

The public paper approach:

    rel_l1 = ||x_t - x_{t-1}||_1 / ||x_{t-1}||_1
    predicted_output_delta = poly(rel_l1)         # calibrated per-model
    accumulated_delta += predicted_output_delta
    if accumulated_delta > rel_l1_thresh:
        run the real model, reset accumulated_delta
    else:
        return previous_output + last_residual    # (or just previous_output)

For a first pass we use `rel_l1` directly (no polynomial). Calibrating the
polynomial for H3 audio + video streams is a Phase-2 task.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

import torch


@dataclasses.dataclass
class TeaCacheState:
    """Rolling state kept across timesteps for a single sampling run."""

    step_idx: int = 0
    prev_input: Optional[torch.Tensor] = None
    prev_output: Optional[torch.Tensor] = None
    accumulated: float = 0.0
    reuse_count: int = 0
    real_count: int = 0

    def reset(self) -> None:
        self.step_idx = 0
        self.prev_input = None
        self.prev_output = None
        self.accumulated = 0.0
        self.reuse_count = 0
        self.real_count = 0


def rel_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    """Relative L1 distance between two tensors, on any device.

    Uses float32 accumulation to avoid bf16/fp16 saturation on large tensors.
    """
    diff = (current.float() - previous.float()).abs().mean()
    ref = previous.float().abs().mean().clamp(min=1e-8)
    return (diff / ref).item()


def should_reuse(
    state: TeaCacheState,
    current_input: torch.Tensor,
    thresh: float,
    start_step: int,
    end_step: int,
    total_steps: int,
) -> bool:
    """Decide whether to reuse the cached output at the current step.

    Guard rails:
      * Never reuse before `start_step` (structure hasn't stabilized).
      * Never reuse in the final `-end_step` steps (fine detail matters).
      * Never reuse on step 0 (nothing cached yet).
      * Never reuse if `prev_output` is None (first real run of the session).
    """
    step = state.step_idx
    if state.prev_output is None or state.prev_input is None:
        return False
    if step < start_step:
        return False
    # Support negative end_step: -N means "last N steps must be real"
    effective_end = end_step if end_step >= 0 else total_steps + end_step
    if step >= effective_end:
        return False

    delta = rel_l1(current_input, state.prev_input)
    state.accumulated += delta
    if state.accumulated < thresh:
        return True
    # Cross the threshold: run real forward, reset accumulator
    state.accumulated = 0.0
    return False


def make_wrapper(
    state: TeaCacheState,
    thresh: float,
    start_step: int,
    end_step: int,
    total_steps: int,
) -> Callable[..., torch.Tensor]:
    """Return a `unet_wrapper_function` compatible with ModelPatcher.

    ComfyUI's `ModelPatcher.set_model_unet_function_wrapper(fn)` calls
    `fn(apply_model, {"input": x, "timestep": t, "c": conds})` on every
    sampling step. We wrap it to decide reuse-vs-real.
    """

    def wrapper(apply_model, args) -> torch.Tensor:
        x = args["input"]

        if should_reuse(state, x, thresh, start_step, end_step, total_steps):
            state.reuse_count += 1
            state.step_idx += 1
            # Reuse previous output. Note: this assumes the model output is a
            # velocity/noise-prediction that changes slowly across steps — true
            # for standard v-prediction / rectified-flow DiTs including H3.
            return state.prev_output

        # Real forward pass
        output = apply_model(x, args["timestep"], **args["c"])
        state.prev_input = x.detach()
        state.prev_output = output.detach()
        state.real_count += 1
        state.step_idx += 1
        return output

    return wrapper
