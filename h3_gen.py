#!/usr/bin/env python3
"""MiniMax H3 视频生成 — 一行命令出片

Usage:
  python h3_gen.py "a red fox walking through a forest" -o fox.mp4
  echo "prompt" | python h3_gen.py --duration 10 --resolution 2K

默认通过 http://127.0.0.1:8188 访问 ComfyUI API。
设置 COMFY_URL 环境变量可指向其他地址。
"""

import argparse, json, os, sys, time, urllib.request, uuid

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")


def api(path, data=None, timeout=600):
    url = f"{COMFY}{path}"
    body = json.dumps(data).encode() if data else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def build_workflow(prompt, *, duration=5, resolution="768P", ratio="16:9",
                   width=None, height=None, steps=20, seed=None,
                   first_frame=None, use_teacache=True, filename_prefix="H3_gen"):
    """Construct an optimized H3 workflow."""
    resolution_map = {
        "768P": (1344, 768),
        "2K": (1920, 1080),
    }

    if width is None and height is None:
        if ratio in resolution_map:
            width, height = resolution_map[ratio]
        else:
            width, height = resolution_map[resolution]

    # Frame count from duration (17k+5 grid at 24fps)
    frames = max(5, duration * 24)
    while frames % 17 != 5:
        frames += 1

    if seed is None:
        seed = int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFFFFFFFFFF

    w = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "5": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0],
            "prompt": prompt,
            "width": width, "height": height, "length": frames}},
        "7": {"class_type": "BasicScheduler", "inputs": {
            "model": ["2", 0], "scheduler": "simple",
            "steps": steps, "denoise": 1.0}},
        "8": {"class_type": "KSamplerSelect", "inputs": {
            "sampler_name": "res_multistep"}},
        "9": {"class_type": "RandomNoise", "inputs": {
            "noise_seed": seed}},
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["9", 0], "guider": ["6", 0],
            "sampler": ["8", 0], "sigmas": ["7", 0],
            "latent_image": ["5", 1]}},
        "11": {"class_type": "VAEDecode", "inputs": {
            "samples": ["10", 0], "vae": ["4", 0]}},
        "12": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "13": {"class_type": "VAEDecodeAudio", "inputs": {
            "samples": ["10", 0], "vae": ["12", 0]}},
        "14": {"class_type": "CreateVideo", "inputs": {
            "images": ["11", 0], "fps": 24, "audio": ["13", 0]}},
        "15": {"class_type": "SaveVideo", "inputs": {
            "video": ["14", 0], "filename_prefix": filename_prefix,
            "format": "mp4", "codec": "auto"}},
    }

    # Optimization chain
    if use_teacache:
        w["16"] = {"class_type": "MiniMaxH3TeaCache", "inputs": {
            "model": ["1", 0],
            "rel_l1_thresh": 0.10, "start_step": 2,
            "end_step": -2, "total_steps": steps}}
        model_output = ["16", 0]
    else:
        model_output = ["1", 0]

    w["2"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {
        "model": model_output, "shift_video": 12.0, "shift_audio": 3.0}}
    w["6"] = {"class_type": "BasicGuider", "inputs": {
        "model": ["2", 0], "conditioning": ["5", 0]}}

    # First frame (image-to-video)
    if first_frame:
        w["5"]["inputs"]["first_frame"] = first_frame  # tensor ref (API mode)

    return w


def main():
    parser = argparse.ArgumentParser(description="MiniMax H3 视频生成")
    parser.add_argument("prompt", nargs="?", help="文本提示词 (也可通过 stdin 传入)")
    parser.add_argument("-o", "--output", default="output/h3_gen.mp4", help="输出路径 (默认: output/h3_gen.mp4)")
    parser.add_argument("-d", "--duration", type=int, default=5, choices=range(4, 16), help="时长(秒), 4-15 (默认: 5)")
    parser.add_argument("-r", "--resolution", default="768P", choices=["768P", "2K"], help="分辨率 (默认: 768P)")
    parser.add_argument("--ratio", default="16:9", choices=["16:9", "9:16", "1:1", "4:3", "3:4"], help="宽高比")
    parser.add_argument("-s", "--steps", type=int, default=20, help="采样步数 (默认: 20)")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--no-teacache", action="store_true", help="禁用 TeaCache 加速")
    parser.add_argument("--dry-run", action="store_true", help="只打印请求 payload，不执行")

    args = parser.parse_args()

    # Prompt from arg or stdin
    prompt = args.prompt
    if not prompt:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        if not prompt:
            parser.error("需要提供 prompt（命令行参数或 stdin）")

    workflow = build_workflow(
        prompt, duration=args.duration, resolution=args.resolution,
        ratio=args.ratio, steps=args.steps, seed=args.seed,
        use_teacache=not args.no_teacache,
        filename_prefix=os.path.splitext(os.path.basename(args.output))[0],
    )

    if args.dry_run:
        print(json.dumps(workflow, indent=2, ensure_ascii=False))
        return

    # Submit
    client_id = str(uuid.uuid4())
    print(f"🎬 提交 H3 生成: {prompt[:80]}...")
    print(f"   分辨率: {workflow['5']['inputs']['width']}×{workflow['5']['inputs']['height']}")
    print(f"   时长: {args.duration}s ({workflow['5']['inputs']['length']} 帧)")
    print(f"   步数: {args.steps}, TeaCache: {not args.no_teacache}")

    start = time.time()
    result = api("/prompt", {"prompt": workflow, "client_id": client_id})
    pid = result["prompt_id"]

    # Poll
    dots = 0
    while True:
        history = api(f"/history/{pid}")
        if pid in history and "outputs" in history[pid]:
            elapsed = time.time() - start
            for node_id, outputs in history[pid]["outputs"].items():
                for media_list in outputs.values():
                    if not isinstance(media_list, list):
                        continue
                    for media in media_list:
                        if "filename" not in media:
                            continue
                        fn = media["filename"]
                        sf = media.get("subfolder", "")
                        url = f"{COMFY}/view?filename={fn}&subfolder={sf}&type=output"
                        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
                        urllib.request.urlretrieve(url, args.output)
                        size = os.path.getsize(args.output)
                        print(f"\n✅ 完成! {elapsed/60:.1f} 分钟 ({elapsed:.0f}s)")
                        print(f"   输出: {args.output} ({size/1024:.0f} KB)")
                        return
            break

        dots += 1
        if dots % 6 == 0:
            elapsed = time.time() - start
            print(f"   ⏳ {elapsed:.0f}s ...")
        elif dots == 1:
            print(f"   ⏳ 生成中...")

        time.sleep(10)


if __name__ == "__main__":
    main()
