"""Benchmark harness for H3 TeaCache.

Talks to a running ComfyUI over HTTP:
  1. Load a base workflow JSON.
  2. Patch it with each (prompt, seed, thresh) triple.
  3. Submit to /prompt, poll /history for completion.
  4. Record wall time and file paths.

Deliberately dependency-free (only urllib + json). Run from repo root:

    python benchmarks/run.py --comfy-url http://hp-z4-server:8188 \
                             --workflow workflows/fl2va-bench.json \
                             --thresholds 0,0.05,0.10,0.15,0.20,0.25 \
                             --output benchmarks/output/

The referenced workflow JSON should contain a `MiniMaxH3TeaCache` node
whose `rel_l1_thresh` widget will be overwritten. Set the value to `0`
to bypass caching (real forward every step).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request


def http_post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def http_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def submit_prompt(base_url: str, workflow: dict) -> str:
    resp = http_post(f"{base_url}/prompt", {"prompt": workflow})
    return resp["prompt_id"]


def wait_for(base_url: str, prompt_id: str, timeout_s: float = 900.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        hist = http_get(f"{base_url}/history/{prompt_id}")
        if prompt_id in hist:
            return hist[prompt_id]
        time.sleep(1.0)
    raise TimeoutError(f"prompt {prompt_id} did not finish in {timeout_s}s")


def find_node(workflow: dict, class_type: str) -> str | None:
    for node_id, node in workflow.items():
        if node.get("class_type") == class_type:
            return node_id
    return None


def patch_prompt(workflow: dict, node_class: str, widget: str, value) -> None:
    node_id = find_node(workflow, node_class)
    if node_id is None:
        raise KeyError(f"no {node_class} node in workflow")
    workflow[node_id]["inputs"][widget] = value


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    p.add_argument("--workflow", required=True, help="ComfyUI API-format workflow JSON")
    p.add_argument("--prompts", default="benchmarks/prompts.jsonl")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--thresholds", default="0,0.05,0.10,0.15,0.20,0.25")
    p.add_argument("--output", default="benchmarks/output")
    args = p.parse_args()

    base_url = args.comfy_url.rstrip("/")
    workflow_template = json.loads(pathlib.Path(args.workflow).read_text())
    prompts = [json.loads(line) for line in pathlib.Path(args.prompts).open()]
    seeds = [int(s) for s in args.seeds.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id", "seed", "thresh", "wall_time_s", "prompt_id_comfy"])
        f.flush()

        for entry in prompts:
            for seed in seeds:
                for thresh in thresholds:
                    wf = copy.deepcopy(workflow_template)
                    # Patch prompt text (assumes the video encode node lives on class
                    # `MiniMaxH3ImageToVideo` or `MiniMaxH3ReferenceToVideo` — adjust
                    # as needed for your benchmarking workflow).
                    for cls in ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"):
                        try:
                            patch_prompt(wf, cls, "prompt", entry["prompt"])
                            break
                        except KeyError:
                            continue

                    patch_prompt(wf, "RandomNoise", "noise_seed", seed)
                    patch_prompt(wf, "MiniMaxH3TeaCache", "rel_l1_thresh", thresh)

                    start = time.perf_counter()
                    pid = submit_prompt(base_url, wf)
                    try:
                        wait_for(base_url, pid, timeout_s=1800.0)
                    except Exception as e:  # noqa: BLE001
                        print(f"[fail] {entry['id']} seed={seed} thresh={thresh}: {e}", file=sys.stderr)
                        writer.writerow([entry["id"], seed, thresh, -1.0, pid])
                        f.flush()
                        continue
                    dt = time.perf_counter() - start
                    print(f"{entry['id']:20s} seed={seed} thresh={thresh:.2f}  {dt:6.1f}s")
                    writer.writerow([entry["id"], seed, thresh, round(dt, 2), pid])
                    f.flush()

    print(f"\nResults written to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
