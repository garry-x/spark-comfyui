#!/usr/bin/env python3
"""LTX-2.3 Text-to-Video Demo #2 - Using dev model with CFG"""
import json, urllib.request, urllib.parse, time, sys, os

COMFYUI_URL = "http://localhost:8188"

def queue_prompt(workflow):
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(f"{COMFYUI_URL}/prompt", data=data)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())

def get_history(prompt_id):
    with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as resp:
        return json.loads(resp.read())

def get_queue():
    with urllib.request.urlopen(f"{COMFYUI_URL}/queue") as resp:
        return json.loads(resp.read())

def build_workflow(prompt_text, seed=123, steps=20, cfg=2.5):
    """Build LTX-2.3 text-to-video workflow with DEV model (supports CFG)."""
    wf = {}
    # Use DEV model for better quality with CFG
    wf["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"}}
    # Gemma3 text encoder
    wf["2"] = {"class_type": "LTXAVTextEncoderLoader", "inputs": {
        "text_encoder": "ltx23_gemma/model_gemma_3_12B_it_fp8_e4m3fn.safetensors",
        "ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors", "device": "default"}}
    # Prompts
    wf["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_text, "clip": ["2", 0]}}
    wf["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "low quality, blurry, distorted, bad anatomy, watermark, text, jpeg artifacts, worst quality", "clip": ["2", 0]}}
    # Conditioning
    wf["5"] = {"class_type": "LTXVConditioning", "inputs": {"positive": ["3", 0], "negative": ["4", 0], "frame_rate": 24.0}}
    # Latent (768x512, ~2 sec)
    wf["6"] = {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 768, "height": 512, "length": 49, "batch_size": 1}}
    # Scheduler
    wf["7"] = {"class_type": "LTXVScheduler", "inputs": {"steps": steps, "max_shift": 2.37, "base_shift": 0.75, "stretch": False, "terminal": 0.1, "latent": ["6", 0]}}
    # Model sampling
    wf["8"] = {"class_type": "ModelSamplingLTXV", "inputs": {"model": ["1", 0], "max_shift": 2.37, "base_shift": 0.75}}
    # Sampler
    wf["9"] = {"class_type": "KSampler", "inputs": {"model": ["8", 0], "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": "euler", "scheduler": "simple", "positive": ["5", 0], "negative": ["5", 1],
        "latent_image": ["6", 0], "denoise": 1.0}}
    # VAE decode
    wf["10"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["9", 0], "vae": ["1", 2],
        "tile_size": 512, "overlap": 64, "temporal_size": 32, "temporal_overlap": 4}}
    # Save
    wf["11"] = {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["10", 0], "filename_prefix": "ltx23_dev_demo",
        "fps": 24.0, "lossless": False, "quality": 90, "method": "default"}}
    return wf

def main():
    print("=" * 60)
    print("LTX-2.3 DEV Model Demo (CFG=2.5, 20 steps)")
    print("=" * 60)
    prompt = "A majestic dragon soaring through snowy mountain peaks, wings spread wide, snow particles swirling in the wind, cinematic lighting, epic fantasy scene"

    wf = build_workflow(prompt, seed=456, steps=20, cfg=2.5)
    print(f"\nPrompt: {prompt}")
    print(f"Nodes: {len(wf)}")

    result = queue_prompt(wf)
    pid = result["prompt_id"]
    print(f"Prompt ID: {pid}")
    print("Generating (this may take 3-5 minutes with dev model + 20 steps)...")

    start = time.time()
    while time.time() - start < 600:
        history = get_history(pid)
        if pid in history:
            h = history[pid]
            status = h.get("status", {})
            if status.get("completed"):
                print(f"\n✅ Completed in {time.time() - start:.0f}s")
                outputs = h.get("outputs", {})
                for nid in outputs:
                    for img in outputs[nid].get("images", []) + outputs[nid].get("gifs", []):
                        fn = img.get("filename", "?")
                        fpath = os.path.join("output", fn)
                        os.makedirs("output", exist_ok=True)
                        url = f"{COMFYUI_URL}/view?filename={urllib.parse.quote(fn)}&type=output"
                        try:
                            urllib.request.urlretrieve(url, fpath)
                            print(f"  📹 {fpath} ({os.path.getsize(fpath)} bytes)")
                        except Exception as e:
                            print(f"  File: {fn} (download error: {e})")
                return 0
            elif status.get("status_str") == "error":
                print(f"\n❌ Error: {status.get('messages', [['?']])[0]}")
                return 1
        time.sleep(5)
        elapsed = time.time() - start
        qi = get_queue()
        running = qi.get("queue_running", [])
        if running:
            print(f"  [{elapsed:.0f}s] Generating...", end="\r")
    print("\nTimeout!")
    return 1

if __name__ == "__main__":
    sys.exit(main())
