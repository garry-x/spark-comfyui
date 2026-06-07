#!/usr/bin/env python3
"""
Export LTX-2.3 workflow as a shareable ComfyUI workflow file.
Saves to user/default/workflows/ and as a downloadable .json file.
"""
import json, os, sys

COMFY_DIR = "/data/ComfyUI/ComfyUI"

def build_ltx23_workflow():
    """Build the LTX-2.3 T2V workflow in ComfyUI API format."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "ltx-2.3-22b-distilled-fp8.safetensors"},
            "_meta": {"title": "Load LTX-2.3 Checkpoint"}
        },
        "2": {
            "class_type": "LTXAVTextEncoderLoader",
            "inputs": {
                "text_encoder": "ltx23_gemma/model_gemma_3_12B_it_fp8_e4m3fn.safetensors",
                "ckpt_name": "ltx-2.3-22b-distilled-fp8.safetensors",
                "device": "default"
            },
            "_meta": {"title": "Load Gemma3 Text Encoder"}
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "A beautiful sunset over a calm ocean, cinematic quality, 4K",
                "clip": ["2", 0]
            },
            "_meta": {"title": "Positive Prompt"}
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "low quality, blurry, distorted, bad anatomy, watermark, worst quality",
                "clip": ["2", 0]
            },
            "_meta": {"title": "Negative Prompt"}
        },
        "5": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["3", 0],
                "negative": ["4", 0],
                "frame_rate": 24.0
            },
            "_meta": {"title": "LTX Conditioning"}
        },
        "6": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": 768,
                "height": 512,
                "length": 49,
                "batch_size": 1
            },
            "_meta": {"title": "Empty Latent (768×512, 2s@24fps)"}
        },
        "7": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": 8,
                "max_shift": 2.37,
                "base_shift": 0.75,
                "stretch": False,
                "terminal": 0.1
            },
            "_meta": {"title": "LTX Scheduler (8 steps)"}
        },
        "8": {
            "class_type": "ModelSamplingLTXV",
            "inputs": {
                "model": ["1", 0],
                "max_shift": 2.37,
                "base_shift": 0.75
            },
            "_meta": {"title": "LTX Model Sampling"}
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["8", 0],
                "seed": 42,
                "steps": 8,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["5", 0],
                "negative": ["5", 1],
                "latent_image": ["6", 0],
                "denoise": 1.0
            },
            "_meta": {"title": "KSampler"}
        },
        "10": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["9", 0],
                "vae": ["1", 2],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 32,
                "temporal_overlap": 4
            },
            "_meta": {"title": "VAE Decode (Tiled)"}
        },
        "11": {
            "class_type": "SaveAnimatedWEBP",
            "inputs": {
                "images": ["10", 0],
                "filename_prefix": "ltx23_output",
                "fps": 24.0,
                "lossless": False,
                "quality": 85,
                "method": "default"
            },
            "_meta": {"title": "Save Animated WebP"}
        }
    }


def build_dev_workflow():
    """DEV model variant with CFG support."""
    wf = build_ltx23_workflow()
    wf["1"]["inputs"]["ckpt_name"] = "ltx-2.3-22b-dev-fp8.safetensors"
    wf["1"]["_meta"]["title"] = "Load LTX-2.3 DEV Checkpoint"
    wf["2"]["inputs"]["ckpt_name"] = "ltx-2.3-22b-dev-fp8.safetensors"
    wf["7"]["inputs"]["steps"] = 20
    wf["7"]["_meta"]["title"] = "LTX Scheduler (20 steps)"
    wf["9"]["inputs"].update({"steps": 20, "cfg": 2.5, "seed": 456})
    wf["9"]["_meta"]["title"] = "KSampler (Dev: CFG=2.5)"
    wf["11"]["inputs"]["filename_prefix"] = "ltx23_dev_output"
    wf["11"]["inputs"]["quality"] = 90
    return wf


def main():
    workflows_dir = os.path.join(COMFY_DIR, "user", "default", "workflows")
    os.makedirs(workflows_dir, exist_ok=True)

    variants = {
        "ltx23_t2v_distilled": {
            "name": "LTX-2.3 Text-to-Video (Distilled, 8-step)",
            "description": "LTX-2.3 视频生成 - 蒸馏模型，8步快速出片，适合预览和快速迭代",
            "workflow": build_ltx23_workflow(),
        },
        "ltx23_t2v_dev": {
            "name": "LTX-2.3 Text-to-Video (DEV, 20-step, CFG)",
            "description": "LTX-2.3 视频生成 - DEV 模型，20步+CFG=2.5，画质最高，支持负向提示词",
            "workflow": build_dev_workflow(),
        },
    }

    for slug, info in variants.items():
        wf = info["workflow"]

        # Save as ComfyUI workflow (API format, used by web UI Load button)
        filepath = os.path.join(workflows_dir, f"{slug}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(wf, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: {filepath}")

        # Also save a pretty copy in project root for sharing
        root_path = os.path.join(COMFY_DIR, f"{slug}.json")
        export = {
            "name": info["name"],
            "description": info["description"],
            "author": "ComfyUI Local",
            "nodes": wf
        }
        with open(root_path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        print(f"✅ Exported: {root_path}")

    print(f"\n📋 工作流已保存到: {workflows_dir}/")
    print("   其他用户可在 ComfyUI Web UI 中通过菜单加载这些工作流")


if __name__ == "__main__":
    main()
