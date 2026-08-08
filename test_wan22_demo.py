#!/usr/bin/env python3
"""Wan 2.2 Text-to-Video Demo with LightX2V 4-step LoRA acceleration."""
import json, urllib.request, urllib.parse, time, sys, os

COMFYUI_URL = "http://localhost:8188"

def qp(wf):
    d=json.dumps({"prompt":wf}).encode()
    with urllib.request.urlopen(urllib.request.Request(f"{COMFYUI_URL}/prompt",data=d)) as r:
        return json.loads(r.read())

def gh(pid):
    with urllib.request.urlopen(f"{COMFYUI_URL}/history/{pid}") as r:
        return json.loads(r.read())

def gq():
    with urllib.request.urlopen(f"{COMFYUI_URL}/queue") as r:
        return json.loads(r.read())

def build_wan22_t2v(prompt, seed=42, steps=20, cfg=6.0, use_lora=True):
    """Wan 2.2 T2V workflow with optional LightX2V LoRA."""
    wf = {}

    # 1: Load diffusion model (high noise for better dynamics)
    wf["1"] = {"class_type": "UNETLoader", "inputs": {
        "unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "weight_dtype": "fp8_e4m3fn"}}

    # 2: Load UMT5 text encoder (type=wan)
    wf["2"] = {"class_type": "CLIPLoader", "inputs": {
        "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "type": "wan"}}

    # 3: Load VAE
    wf["3"] = {"class_type": "VAELoader", "inputs": {
        "vae_name": "wan_2.1_vae.safetensors"}}

    # 4: Positive prompt
    wf["4"] = {"class_type": "CLIPTextEncode", "inputs": {
        "text": prompt, "clip": ["2", 0]}}

    # 5: Negative prompt
    wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {
        "text": "low quality, blurry, distorted, bad anatomy, watermark, text, jpeg artifacts, worst quality, static",
        "clip": ["2", 0]}}

    # 6: Wan conditioning + latent (T2V mode without start_image)
    wf["6"] = {"class_type": "WanImageToVideo", "inputs": {
        "positive": ["4", 0], "negative": ["5", 0], "vae": ["3", 0],
        "width": 832, "height": 480, "length": 81, "batch_size": 1}}

    # 7: LoRA (optional, LightX2V 4-step acceleration)
    if use_lora:
        wf["7"] = {"class_type": "LoraLoader", "inputs": {
            "model": ["1", 0],
            "clip": ["2", 0],
            "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
            "strength_model": 1.0, "strength_clip": 1.0}}
        model_src = ["7", 0]
    else:
        model_src = ["1", 0]

    # 8: KSampler
    wf["8"] = {"class_type": "KSampler", "inputs": {
        "model": model_src if use_lora else ["1", 0],
        "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": "uni_pc", "scheduler": "simple",
        "positive": ["6", 0], "negative": ["6", 1],
        "latent_image": ["6", 2], "denoise": 1.0}}

    # 9: VAE decode
    wf["9"] = {"class_type": "VAEDecodeTiled", "inputs": {
        "samples": ["8", 0], "vae": ["3", 0],
        "tile_size": 512, "overlap": 64,
        "temporal_size": 32, "temporal_overlap": 4}}

    # 10: Save
    wf["10"] = {"class_type": "SaveAnimatedWEBP", "inputs": {
        "images": ["9", 0], "filename_prefix": "wan22_demo",
        "fps": 16.0, "lossless": False, "quality": 85, "method": "default"}}

    return wf

def main():
    print("="*60)
    print("Wan 2.2 T2V Demo (LightX2V 4-step LoRA)")
    print("="*60)

    prompt = "A majestic dragon soaring through snowy mountain peaks, wings spread wide, snow particles swirling in the wind, cinematic lighting, epic fantasy scene, 4K quality"

    wf = build_wan22_t2v(prompt, seed=42, steps=4, cfg=2.0, use_lora=True)
    print(f"Prompt: {prompt}")
    print(f"Nodes: {len(wf)}, Steps: 4 (LoRA accelerated), CFG: 2.0")

    result = qp(wf)
    pid = result["prompt_id"]
    print(f"Prompt ID: {pid}")
    print("Generating...")

    start = time.time()
    while time.time()-start < 600:
        h = gh(pid)
        if pid in h:
            s = h[pid].get("status",{})
            if s.get("completed"):
                print(f"\n✅ Done ({time.time()-start:.0f}s)")
                for nid,out in h[pid].get("outputs",{}).items():
                    for img in out.get("images",[])+out.get("gifs",[]):
                        fn=img["filename"]
                        fp=os.path.join("output",fn)
                        os.makedirs("output",exist_ok=True)
                        url=f"{COMFYUI_URL}/view?filename={urllib.parse.quote(fn)}&type=output"
                        urllib.request.urlretrieve(url,fp)
                        print(f"  📹 {fp} ({os.path.getsize(fp)} bytes)")
                return 0
            elif s.get("status_str")=="error":
                msgs=s.get("messages",[["?"]])
                print(f"\n❌ {msgs[0]}")
                return 1
        time.sleep(5)
        print(f"  [{time.time()-start:.0f}s]", end="\r")
    print("\nTimeout")
    return 1

if __name__=="__main__":
    sys.exit(main())
