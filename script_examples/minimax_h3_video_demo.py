"""Create and download a MiniMax H3 text-to-video demo.

Usage:
    MINIMAX_API_KEY=... python script_examples/minimax_h3_video_demo.py
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from urllib.request import Request, urlopen


API_BASE_URL = "https://api.minimax.io"


def request_json(url: str, api_key: str, data: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(data).encode("utf-8")
    else:
        body = None

    with urlopen(Request(url, data=body, headers=headers), timeout=60) as response:
        return json.load(response)


def create_video(api_key: str, prompt: str, duration: int, resolution: str, ratio: str) -> str:
    payload = {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": prompt}],
        "duration": duration,
        "resolution": resolution,
        "ratio": ratio,
    }
    response = request_json(f"{API_BASE_URL}/v2/video_generation", api_key, payload)
    return response["task_id"]


def wait_for_video(api_key: str, task_id: str) -> str:
    while True:
        response = request_json(
            f"{API_BASE_URL}/v2/query/video_generation/{task_id}", api_key
        )
        task = response["task"]
        status = task["status"]
        print(f"MiniMax H3 task {task_id}: {status}")
        if status == "succeeded":
            return task["content"]["url"]
        if status in {"failed", "cancelled"}:
            raise RuntimeError(f"MiniMax H3 generation failed: {task.get('error')}")
        time.sleep(10)


def download_video(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response, output_path.open("wb") as output:
        shutil.copyfileobj(response, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a MiniMax H3 text-to-video demo.")
    parser.add_argument(
        "--prompt",
        default="A red paper boat sails across a sunlit pond, gentle ripples, cinematic wide shot.",
    )
    parser.add_argument("--duration", type=int, default=5, choices=range(4, 16))
    parser.add_argument("--resolution", choices=["768P", "2K"], default="768P")
    parser.add_argument("--ratio", choices=["16:9", "9:16", "1:1", "4:3", "3:4"], default="16:9")
    parser.add_argument("--output", type=Path, default=Path("output/minimax_h3_demo.mp4"))
    parser.add_argument("--dry-run", action="store_true", help="Print the request payload without calling MiniMax.")
    args = parser.parse_args()

    payload = {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": args.prompt}],
        "duration": args.duration,
        "resolution": args.resolution,
        "ratio": args.ratio,
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise SystemExit("Set MINIMAX_API_KEY before running this demo.")

    task_id = create_video(api_key, args.prompt, args.duration, args.resolution, args.ratio)
    print(f"Created MiniMax H3 task: {task_id}")
    video_url = wait_for_video(api_key, task_id)
    download_video(video_url, args.output)
    print(f"Saved video to {args.output}")


if __name__ == "__main__":
    main()
