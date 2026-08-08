#!/usr/bin/env python3
"""
Wan 2.2 T2V 高清优化版
改进: 双阶段(HighNoise→LowNoise), dpmpp_2m+sgm_uniform, 更多步数, 更高CFG
"""
import json, urllib.request, urllib.parse, time, sys, os

url = "http://localhost:8188"

def submit(wf):
    d = json.dumps({"prompt": wf}).encode()
    with urllib.request.urlopen(urllib.request.Request(f"{url}/prompt", data=d)) as r:
        return json.loads(r.read())["prompt_id"]

def wait(pid, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        with urllib.request.urlopen(f"{url}/history/{pid}") as r:
            h = json.loads(r.read())
        if pid in h:
            s = h[pid]["status"]
            if s["completed"]:
                return h[pid]
            if s.get("status_str") == "error":
                print(f"❌ {s.get('messages', ['?'])}")
                return None
        print(f"  [{time.time()-start:.0f}s]", end="\r")
        time.sleep(5)
    return None

def build_hq_workflow(prompt, seed=42, steps=15, cfg_high=5.0, cfg_low=3.0):
    """双阶段 Wan 2.2 T2V 工作流：高噪声→低噪声"""
    wf = {}
    nid = 0

    # === LOADERS ===
    wf["1"] = {"class_type": "UNETLoader", "inputs": {
        "unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "weight_dtype": "fp8_e4m3fn"}}
    wf["2"] = {"class_type": "UNETLoader", "inputs": {
        "unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        "weight_dtype": "fp8_e4m3fn"}}
    wf["3"] = {"class_type": "CLIPLoader", "inputs": {
        "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}}
    wf["4"] = {"class_type": "VAELoader", "inputs": {
        "vae_name": "wan_2.1_vae.safetensors"}}

    # === PROMPTS ===
    wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {
        "text": prompt, "clip": ["3", 0]}}
    wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {
        "text": "low quality, blurry, out of focus, soft, fuzzy, lowres, pixelated, distorted, bad anatomy, watermark, text, jpeg artifacts, worst quality, static, ugly, cropped",
        "clip": ["3", 0]}}

    # === STAGE 1: HIGH NOISE (structure + motion) ===
    # Lightning LoRA on high-noise model
    wf["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["1", 0], "clip": ["3", 0],
        "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
        "strength_model": 1.0, "strength_clip": 1.0}}

    # Conditioning + latent for stage 1
    wf["8"] = {"class_type": "WanImageToVideo", "inputs": {
        "positive": ["5", 0], "negative": ["6", 0], "vae": ["4", 0],
        "width": 832, "height": 480, "length": 81, "batch_size": 1}}

    half_steps = steps // 2
    wf["9"] = {"class_type": "KSamplerAdvanced", "inputs": {
        "model": ["7", 0], "add_noise": "enable", "noise_seed": seed,
        "steps": steps, "cfg": cfg_high,
        "sampler_name": "dpmpp_2m", "scheduler": "sgm_uniform",
        "positive": ["8", 0], "negative": ["8", 1],
        "latent_image": ["8", 2],
        "start_at_step": 0, "end_at_step": half_steps,
        "return_with_leftover_noise": "enable"}}

    # === STAGE 2: LOW NOISE (detail + sharpness) ===
    # Lightning LoRA on low-noise model
    wf["10"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["2", 0], "clip": ["3", 0],
        "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
        "strength_model": 1.0, "strength_clip": 1.0}}

    wf["11"] = {"class_type": "KSamplerAdvanced", "inputs": {
        "model": ["10", 0], "add_noise": "enable", "noise_seed": seed,
        "steps": steps, "cfg": cfg_low,
        "sampler_name": "dpmpp_2m", "scheduler": "sgm_uniform",
        "positive": ["8", 0], "negative": ["8", 1],
        "latent_image": ["9", 0],
        "start_at_step": half_steps, "end_at_step": steps,
        "return_with_leftover_noise": "disable"}}

    # === DECODE + SAVE ===
    wf["12"] = {"class_type": "VAEDecodeTiled", "inputs": {
        "samples": ["11", 0], "vae": ["4", 0],
        "tile_size": 512, "overlap": 64, "temporal_size": 32, "temporal_overlap": 4}}

    wf["13"] = {"class_type": "SaveAnimatedWEBP", "inputs": {
        "images": ["12", 0], "filename_prefix": "wan22_hq",
        "fps": 16.0, "lossless": False, "quality": 90, "method": "default"}}

    return wf

def main():
    print("=" * 60)
    print("Wan 2.2 高清优化版 (双阶段 dpmpp_2m)")
    print("=" * 60)

    prompt = "A cute orange tabby kitten running playfully across a cozy sunlit living room floor, chasing a red ball, warm afternoon light streaming through curtains, wooden floor with soft rug, green plants near window, smooth cinematic motion, high quality, sharp, detailed, masterpiece, 8K"

    print(f"Prompt: {prompt[:80]}...")
    print(f"Steps: 15 (high:8 + low:7), CFG: 5.0→3.0")
    print(f"Sampler: dpmpp_2m + sgm_uniform")

    wf = build_hq_workflow(prompt, seed=42, steps=15, cfg_high=5.0, cfg_low=3.0)
    print(f"Nodes: {len(wf)}")
    for nid in sorted(wf, key=int):
        print(f"  [{nid}] {wf[nid]['class_type']}")

    pid = submit(wf)
    print(f"\nPID: {pid}")
    print("Generating (may take 4-6 minutes)...")

    result = wait(pid)
    if result:
        outputs = result.get("outputs", {})
        for nid in outputs:
            for img in outputs[nid].get("images", []) + outputs[nid].get("gifs", []):
                fn = img["filename"]
                src = f"/data/ComfyUI/ComfyUI/output/{fn}"
                os.system(f"cp {src} /data/ComfyUI/ComfyUI/ 2>/dev/null")
                print(f"✅ {fn} (output/)")

if __name__ == "__main__":
    main()
