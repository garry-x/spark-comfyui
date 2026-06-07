#!/usr/bin/env python3
"""
LTX-2.3 Text-to-Video Demo
Uses ComfyUI API to generate a video from a text prompt.
"""
import json
import urllib.request
import urllib.parse
import time
import sys
import os
import uuid

COMFYUI_URL = "http://localhost:8188"

def queue_prompt(prompt_workflow):
    """Submit a workflow to ComfyUI and return the prompt_id."""
    data = json.dumps({"prompt": prompt_workflow}).encode("utf-8")
    req = urllib.request.Request(f"{COMFYUI_URL}/prompt", data=data)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"ERROR submitting prompt: {e}")
        if hasattr(e, 'read'):
            print(e.read().decode())
        return None

def get_history(prompt_id):
    """Get the execution history for a prompt."""
    with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as resp:
        return json.loads(resp.read())

def get_queue():
    """Get current queue status."""
    with urllib.request.urlopen(f"{COMFYUI_URL}/queue") as resp:
        return json.loads(resp.read())

def wait_for_prompt(prompt_id, timeout=600):
    """Wait for a prompt to finish executing."""
    start = time.time()
    while time.time() - start < timeout:
        history = get_history(prompt_id)
        if prompt_id in history:
            return history[prompt_id]

        queue_info = get_queue()
        running = queue_info.get("queue_running", [])
        pending = queue_info.get("queue_pending", [])

        for item in running:
            if item[1] == prompt_id:
                print(f"  Running... ({time.time() - start:.0f}s elapsed)", end="\r")
                break

        if not running and not pending:
            time.sleep(2)
            history = get_history(prompt_id)
            if prompt_id in history:
                return history[prompt_id]

        time.sleep(3)

    raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")

def download_output(filename, output_dir="./output"):
    """Download a generated file from ComfyUI."""
    os.makedirs(output_dir, exist_ok=True)
    local_path = os.path.join(output_dir, os.path.basename(filename))

    url = f"{COMFYUI_URL}/view?filename={urllib.parse.quote(filename)}&type=output"
    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"  Downloaded: {local_path} ({os.path.getsize(local_path)} bytes)")
        return local_path
    except Exception as e:
        print(f"  Download error: {e}")
        return None

def build_workflow():
    """
    Build an LTX-2.3 text-to-video workflow.

    Node chain:
    1. CheckpointLoaderSimple  → MODEL, CLIP, VAE
    2. CLIPTextEncode (+)      → positive conditioning
    3. CLIPTextEncode (-)      → negative conditioning
    4. LTXVConditioning        → conditioning with frame_rate
    5. EmptyLTXVLatentVideo    → video latent
    6. LTXVScheduler           → sigmas
    7. KSamplerSelect          → sampler name
    8. ModelSamplingLTXV       → model with LTX sampling
    9. KSampler                → denoised latent
   10. VAEDecodeTiled          → video frames (IMAGE)
   11. SaveAnimatedWEBP        → animated webp output
    """

    wf = {}

    # Node 1: Load the LTX-2.3 checkpoint (model + VAE; CLIP will be None for transformer-only)
    wf["1"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "ltx-2.3-22b-distilled-fp8.safetensors"
        }
    }

    # Node 2: Load Gemma3 text encoder + text projection (LTXAV-specific loader)
    wf["2"] = {
        "class_type": "LTXAVTextEncoderLoader",
        "inputs": {
            "text_encoder": "ltx23_gemma/model_gemma_3_12B_it_fp8_e4m3fn.safetensors",
            "ckpt_name": "ltx-2.3-22b-distilled-fp8.safetensors",
            "device": "default"
        }
    }

    # Node 3: Positive prompt (use Gemma3-generated prompt for better quality)
    wf["3"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "A beautiful sunset over a calm ocean, waves gently lapping at the shore, golden light reflecting on the water, cinematic quality, 4K",
            "clip": ["2", 0]  # CLIP from LTXAVTextEncoderLoader
        }
    }

    # Node 4: Negative prompt
    wf["4"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "low quality, blurry, distorted, bad anatomy, watermark, text, jpeg artifacts, worst quality, static",
            "clip": ["2", 0]
        }
    }

    # Node 5: LTX conditioning (wraps conditioning with frame_rate)
    wf["5"] = {
        "class_type": "LTXVConditioning",
        "inputs": {
            "positive": ["3", 0],
            "negative": ["4", 0],
            "frame_rate": 24.0
        }
    }

    # Node 6: Empty video latent (768x512, 49 frames ~2 seconds at 24fps)
    wf["6"] = {
        "class_type": "EmptyLTXVLatentVideo",
        "inputs": {
            "width": 768,
            "height": 512,
            "length": 49,
            "batch_size": 1
        }
    }

    # Node 7: LTX scheduler
    wf["7"] = {
        "class_type": "LTXVScheduler",
        "inputs": {
            "steps": 8,
            "max_shift": 2.37,
            "base_shift": 0.75,
            "stretch": False,
            "terminal": 0.1
        }
    }

    # Node 8: Model sampling for LTX
    wf["8"] = {
        "class_type": "ModelSamplingLTXV",
        "inputs": {
            "model": ["1", 0],
            "max_shift": 2.37,
            "base_shift": 0.75
        }
    }

    # Node 9: KSampler
    wf["9"] = {
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
        }
    }

    # Node 10: VAE Decode (tiled for video)
    wf["10"] = {
        "class_type": "VAEDecodeTiled",
        "inputs": {
            "samples": ["9", 0],
            "vae": ["1", 2],
            "tile_size": 512,
            "overlap": 64,
            "temporal_size": 32,
            "temporal_overlap": 4
        }
    }

    # Node 11: Save as animated WEBP
    wf["11"] = {
        "class_type": "SaveAnimatedWEBP",
        "inputs": {
            "images": ["10", 0],
            "filename_prefix": "ltx23_demo",
            "fps": 24.0,
            "lossless": False,
            "quality": 85,
            "method": "default"
        }
    }

    return wf


def main():
    print("=" * 60)
    print("LTX-2.3 Text-to-Video Demo")
    print("=" * 60)

    print("\n[1] Building workflow...")
    workflow = build_workflow()
    print(f"    Nodes: {len(workflow)}")
    for node_id, node in workflow.items():
        print(f"    [{node_id}] {node['class_type']}")

    print("\n[2] Submitting to ComfyUI...")
    result = queue_prompt(workflow)
    if not result:
        print("    FAILED to submit workflow!")
        return 1

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        print(f"    Unexpected response: {result}")
        return 1

    print(f"    Prompt ID: {prompt_id}")

    print("\n[3] Waiting for generation to complete...")
    print("    (This may take several minutes for video generation)")
    try:
        history = wait_for_prompt(prompt_id, timeout=600)
    except TimeoutError:
        print("\n    TIMEOUT - check ComfyUI logs for errors")
        return 1

    print(f"\n    Completed! Status: {history.get('status', {})}")

    # Check for outputs
    outputs = history.get("outputs", {})
    print(f"\n[4] Output files:")
    found = False
    for node_id, node_output in outputs.items():
        images = node_output.get("images", [])
        gifs = node_output.get("gifs", [])
        for img in images + gifs:
            filename = img.get("filename", "")
            subfolder = img.get("subfolder", "")
            full_path = os.path.join(subfolder, filename) if subfolder else filename
            print(f"    Node {node_id}: {full_path} ({img.get('type', '?')})")
            download_output(full_path)
            found = True

    if not found:
        # Check for errors
        status = history.get("status", {})
        if status.get("status_str") == "error":
            print(f"    ERROR: {status.get('messages', [['Unknown error']])[0]}")
            return 1
        print("    No output files found. Check ComfyUI logs.")
        return 1

    print("\n" + "=" * 60)
    print("Demo complete! Check ./output/ directory for results.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
