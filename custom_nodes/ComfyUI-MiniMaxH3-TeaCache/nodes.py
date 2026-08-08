"""ComfyUI node definitions."""

from __future__ import annotations

from .teacache import TeaCacheState, make_wrapper


class MiniMaxH3TeaCacheNode:
    """Wrap a MiniMax H3 MODEL so that TeaCache decides reuse-vs-real per step.

    Insert between UNETLoader and the guider:

        UNETLoader -> MiniMax H3 TeaCache -> CFGGuider / BasicGuider
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rel_l1_thresh": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": (
                            "Reuse cached output while accumulated rel-L1 input delta "
                            "stays below this. Higher = more speedup, more drift."
                        ),
                    },
                ),
                "start_step": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 100,
                        "tooltip": "First step at which caching is allowed. Early steps set structure — keep them real.",
                    },
                ),
                "end_step": (
                    "INT",
                    {
                        "default": -2,
                        "min": -20,
                        "max": 100,
                        "tooltip": "Last step at which caching is allowed. Negative means 'this many steps from the end'.",
                    },
                ),
                "total_steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 200,
                        "tooltip": "Total number of sampling steps used by the sampler. Must match the scheduler.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/model"
    TITLE = "MiniMax H3 TeaCache"

    def apply(self, model, rel_l1_thresh, start_step, end_step, total_steps):
        # Fresh state for every graph execution (ComfyUI re-runs on every prompt).
        state = TeaCacheState()
        state.reset()

        wrapper = make_wrapper(
            state=state,
            thresh=rel_l1_thresh,
            start_step=start_step,
            end_step=end_step,
            total_steps=total_steps,
        )

        # Clone so we don't mutate the shared cached model.
        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)

        # Stash state so a companion "Stats" node (future) can read reuse/real counts.
        patched.model_options.setdefault("teacache", {})["state"] = state
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TeaCache": MiniMaxH3TeaCacheNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TeaCache": "MiniMax H3 TeaCache",
}
