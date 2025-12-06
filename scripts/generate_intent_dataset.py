# scripts/generate_intent_dataset.py
"""
Generate a prompt/response dataset from historical telemetry logs.

Reads telemetry JSONL files (as written by server/server.py), samples every N
seconds per vehicle, and produces a JSONL where each entry contains:
  - ts, veh_id
  - telemetry snapshot for that vehicle
  - prompt messages (system/user) matching the vehicular agent prompt
  - optional OpenAI response (if --call-api is set and OPENAI_API_KEY is present)

Images: if a top-down JPEG exists at <session_dir>/frames_top/frame_<...>.jpg,
you can pass --top-frame to attach that path; otherwise the script skips the
image. Historical logs often won’t have aligned frames, so image_b64 may be
null.

Usage examples:
  python scripts/generate_intent_dataset.py \
    --logs data/logs/20241201_120000/telemetry.jsonl \
    --output dataset.jsonl \
    --sample-sec 10 \
    --model gpt-4o-mini

  # Call OpenAI to include model outputs (requires OPENAI_API_KEY env)
  python scripts/generate_intent_dataset.py --logs data/logs/*/telemetry.jsonl --call-api
"""

import argparse
import base64
import glob
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple

import httpx


SYSTEM_PROMPT = (
    "Vehicular agent: output only compact JSON for one car. "
    "Use at most 3 steps to satisfy the goal. "
    'Allowed actions: "lane_left","lane_right","set_speed","speed_up","speed_down","hold","brake". '
    'If another vehicle blocks or is too close, include a request (to ["veh_id"| "unknown"]) such as "slow down" or "yield". '
    'JSON shape: {"ego_veh_id":n,"plan_summary":"...","plan_steps":[{"id":"s1","description":"...","action":"lane_left","target_lane_id":-1,"target_speed_kmh":40}], "ego_action":"lane left","reason":"...","confidence":0.6,"request":{"to":["150"],"ask":"slow down"}}'
)

FORMAT_REMINDER = (
    'Respond with ONLY minified JSON. Example: {"ego_veh_id":101,"plan_summary":"Shift left and match flow","plan_steps":[{"id":"s1","description":"Move left to faster lane","action":"lane_left","target_lane_id":-2},{"id":"s2","description":"Hold 45 km/h","action":"set_speed","target_speed_kmh":45}],"ego_action":"lane left","reason":"avoid slow traffic","confidence":0.78,"request":{"to":["150"],"ask":"slow down"}}'
)


def load_logs(paths: Iterable[str]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            continue


def sample_entries(log_iter: Iterable[Dict[str, Any]], sample_sec: float) -> Iterable[Tuple[float, Dict[str, Any]]]:
    last_ts: Dict[str, float] = {}
    for obj in log_iter:
        ts = obj.get("t") or obj.get("ts")
        vehicles = obj.get("vehicles") or obj.get("payload")
        if ts is None or not isinstance(vehicles, dict):
            continue
        for vid_str, tele in vehicles.items():
            try:
                vid_key = str(int(vid_str))
            except Exception:
                continue
            prev = last_ts.get(vid_key, 0)
            if prev and (ts - prev) < sample_sec:
                continue
            last_ts[vid_key] = ts
            yield ts, {"veh_id": int(vid_key), **tele}


def maybe_encode_image(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    except Exception:
        return ""


def find_top_frame(top_dir: str, veh_id: int, index: int | None = None) -> str:
    """Try patterns including veh_id and optional index to locate a top frame."""
    patterns = []
    if index is not None:
        patterns.append(f"veh_{veh_id}_top_{index:06d}.jpg")
    patterns.extend([
        f"frame_top_{veh_id}.jpg",
        f"top_{veh_id}.jpg",
        f"frame_{veh_id}.jpg",
        f"{veh_id}.jpg",
    ])
    for name in patterns:
        path = os.path.join(top_dir, name)
        if os.path.isfile(path):
            return path

    # Fallback: glob for any veh-specific file
    glob_candidates = list(Path(top_dir).glob(f"veh_{veh_id}_top_*.jpg"))
    if glob_candidates:
        # pick the latest (lexicographically) to approximate most recent frame
        return str(sorted(glob_candidates)[-1])
    return ""


def build_messages(goal: str, tele: Dict[str, Any], top_b64: str) -> list:
    user_text = (
        f"goal: {goal}\n"
        f"speed_kmh: {tele.get('speed_kmh', 0)}\n"
        f"lane_id: {tele.get('lane_id')}\n"
        f"veh_id: {tele.get('veh_id')}"
    )
    content = [{"type": "text", "text": user_text}]
    if top_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{top_b64}"}})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
        {"role": "user", "content": FORMAT_REMINDER},
    ]


async def call_openai(messages: list, model: str, api_key: str) -> Dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
    resp.raise_for_status()
    return resp.json()


async def main():
    parser = argparse.ArgumentParser(description="Generate intent dataset from telemetry logs")
    parser.add_argument("--logs", nargs="+", help="Paths/globs to telemetry JSONL files", required=True)
    parser.add_argument("--output", default="intent_dataset.jsonl", help="Output JSONL path")
    parser.add_argument("--sample-sec", type=float, default=10.0, help="Sampling interval per vehicle")
    parser.add_argument("--goal", default="Drive safely and keep lane discipline.", help="Default goal text")
    parser.add_argument("--model", default=os.getenv("VITE_OPENAI_MODEL", "gpt-4o-mini"), help="Model name")
    parser.add_argument("--call-api", action="store_true", help="Call OpenAI and include responses")
    parser.add_argument("--top-frame", help="Optional template for top frame path, use {veh_id} placeholder")
    parser.add_argument("--top-dir", help="Optional directory of top frames; script will try common patterns using veh_id")
    parser.add_argument("--include-image-b64", action="store_true", help="Store base64 image in each record (requires --top-frame)")
    parser.add_argument("--batch-ndjson", help="Optional path to write OpenAI Batch API NDJSON payload")
    parser.add_argument("--batch-max-tokens", type=int, default=512, help="Max completion tokens per batch request body")
    parser.add_argument("--cost-rate", type=float, default=0.0, help="Optional $ per 1K tokens (total usage logged)")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY") or ""
    if args.call_api and not api_key:
        raise SystemExit("--call-api set but OPENAI_API_KEY is missing")

    log_paths = []
    for pattern in args.logs:
        log_paths.extend(glob.glob(pattern))
    if not log_paths:
        raise SystemExit("No telemetry logs found for given paths")

    entries = sample_entries(load_logs(log_paths), args.sample_sec)
    written = 0
    total_input_tokens = 0
    total_output_tokens = 0
    batch_fp = None
    veh_frame_counter: Dict[int, int] = {}
    try:
        if args.batch_ndjson:
            batch_fp = open(args.batch_ndjson, "w", encoding="utf-8")
    except Exception as exc:
        raise SystemExit(f"Failed to open batch NDJSON: {exc}")
    with open(args.output, "w", encoding="utf-8") as out:
        for ts, tele in entries:
            top_b64 = ""
            vid = tele.get("veh_id")
            frame_idx = veh_frame_counter.get(vid, 0)

            if args.top_frame:
                candidate = args.top_frame.format(veh_id=vid, index=frame_idx)
                top_b64 = maybe_encode_image(candidate)
            elif args.top_dir:
                candidate = find_top_frame(args.top_dir, vid, index=frame_idx)
                top_b64 = maybe_encode_image(candidate)
            if (args.top_frame or args.top_dir) and not top_b64:
                # skip samples without an image when frames are required
                continue

            messages = build_messages(args.goal, tele, top_b64)
            record = {
                "ts": ts,
                "veh_id": tele.get("veh_id"),
                "goal": args.goal,
                "telemetry": tele,
                "messages": messages,
            }
            if args.include_image_b64 and top_b64:
                record["image_b64"] = top_b64
            if args.call_api:
                try:
                    resp = await call_openai(messages, args.model, api_key)
                    record["response"] = resp
                    usage = resp.get("usage") or {}
                    total_input_tokens += usage.get("prompt_tokens", 0)
                    total_output_tokens += usage.get("completion_tokens", 0)
                except Exception as e:
                    record["response_error"] = str(e)
            if batch_fp:
                req = {
                    "custom_id": f"veh-{tele.get('veh_id')}-{int(ts * 1000)}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": messages,
                        "response_format": {"type": "json_object"},
                        "max_completion_tokens": args.batch_max_tokens,
                    },
                }
                batch_fp.write(json.dumps(req, separators=(",", ":")) + "\n")
            out.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
            written += 1
            veh_frame_counter[vid] = frame_idx + 1

    print(f"Wrote {written} rows to {args.output}")
    if batch_fp:
        batch_fp.close()
        print(f"Wrote batch NDJSON to {args.batch_ndjson} (lines match dataset rows)")
    if args.call_api:
        cost = 0.0
        if args.cost_rate > 0:
            total_tokens = total_input_tokens + total_output_tokens
            cost = (total_tokens / 1000.0) * args.cost_rate
        print(
            f"Token usage — prompt: {total_input_tokens}, completion: {total_output_tokens}, "
            f"cost (approx): ${cost:.4f}" if args.cost_rate else
            f"Token usage — prompt: {total_input_tokens}, completion: {total_output_tokens}"
        )


if __name__ == "__main__":
    try:
        import anyio

        anyio.run(main)
    except ImportError:
        import asyncio

        asyncio.run(main())
