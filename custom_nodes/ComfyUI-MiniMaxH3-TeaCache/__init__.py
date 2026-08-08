"""ComfyUI-MiniMaxH3-TeaCache

Timestep-cache acceleration for MiniMax H3 video generation.
Wraps the model's unet function via ModelPatcher — no core file modification.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
