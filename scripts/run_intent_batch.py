# scripts/run_intent_batch.py
"""
End-to-end helper to:
  1) Load .env for OPENAI_API_KEY (and model override).
  2) Generate a batch NDJSON of vehicular prompts from telemetry logs.
  3) Upload the NDJSON to OpenAI /files.
  4) Create a Batch job targeting /v1/chat/completions.
  5) Poll until completion and download results.
  6) Merge responses back into a JSONL dataset.

Note: This script makes live network calls to OpenAI. Use only if you intend
to run it locally with your key. It assumes the Batch API supports direct
file upload + batch creation. For large datasets, mind file size limits.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, Any

import httpx
from dotenv import load_dotenv

from generate_intent_dataset import (
    build_messages,
    load_logs,
    sample_entries,
    maybe_encode_image,
    find_top_frame,
)


BATCH_POLL_INTERVAL = 5.0  # seconds
FILES_ENDPOINT = "https://api.openai.com/v1/files"
BATCH_ENDPOINT = "https://api.openai.com/v1/batches"


def load_env():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def make_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
    }


async def upload_file(ndjson_path: Path, api_key: str) -> str:
    headers = make_headers(api_key)
    files = {"file": (ndjson_path.name, ndjson_path.open("rb"), "application/json")}
    data = {"purpose": "batch"}
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(FILES_ENDPOINT, headers=headers, data=data, files=files)
    resp.raise_for_status()
    return resp.json()["id"]


async def create_batch(file_id: str, api_key: str) -> str:
    headers = make_headers(api_key)
    payload = {
        "input_file_id": file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(BATCH_ENDPOINT, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["id"]


async def poll_batch(batch_id: str, api_key: str):
    headers = make_headers(api_key)
    async with httpx.AsyncClient(timeout=None) as client:
        while True:
            resp = await client.get(f"{BATCH_ENDPOINT}/{batch_id}", headers=headers)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")
            if status in ("completed", "failed", "expired"):
                return data
            await asyncio.sleep(BATCH_POLL_INTERVAL)


async def download_batch_output(batch_id: str, api_key: str, output_path: Path):
    headers = make_headers(api_key)
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.get(f"{BATCH_ENDPOINT}/{batch_id}/output", headers=headers)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)


def merge_responses(dataset_path: Path, batch_output_path: Path, merged_path: Path):
    # Batch output NDJSON includes "custom_id" and "response" per line
    resp_map = {}
    with batch_output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id")
            if cid:
                resp_map[cid] = obj

    with dataset_path.open("r", encoding="utf-8") as src, merged_path.open("w", encoding="utf-8") as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("custom_id")
            if cid and cid in resp_map:
                rec["response"] = resp_map[cid].get("response") or resp_map[cid].get("body")
            out.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")


async def main():
    load_env()
    parser = argparse.ArgumentParser(description="Run OpenAI Batch for intent generation and merge results")
    parser.add_argument("--logs", nargs="+", required=True, help="Paths/globs to telemetry JSONL files")
    parser.add_argument("--output", default="intent_dataset.jsonl", help="Output JSONL (prompts only)")
    parser.add_argument("--merged-output", default="intent_dataset_with_responses.jsonl", help="Merged JSONL with responses")
    parser.add_argument("--ndjson", default="intent_batch.ndjson", help="Batch NDJSON payload path")
    parser.add_argument("--batch-output", default="intent_batch_output.ndjson", help="Downloaded batch output NDJSON path")
    parser.add_argument("--sample-sec", type=float, default=10.0, help="Sampling interval per vehicle")
    parser.add_argument("--goal", default="Drive safely and keep lane discipline.", help="Default goal text")
    parser.add_argument("--model", default=os.getenv("VITE_OPENAI_MODEL", "gpt-4o-mini"), help="Model name")
    parser.add_argument("--top-dir", help="Directory of top frames; tries common veh_id patterns. If omitted, will infer from log parent session folder (../frames_top)")
    parser.add_argument("--top-frame", help="Optional template with {veh_id} (e.g., data/logs/<session>/frames_top/veh_{veh_id}_top_{index:06d}.jpg)")
    parser.add_argument("--include-image-b64", action="store_true", help="Store base64 image in records")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY missing in environment")

    # Build dataset + batch NDJSON
    log_paths = []
    for pattern in args.logs:
        log_paths.extend(Path().glob(pattern))

    # If a directory is provided, include all telemetry.jsonl under it
    expanded = []
    for p in log_paths:
        if p.is_dir():
            expanded.extend(p.rglob("telemetry.jsonl"))
        else:
            expanded.append(p)
    log_paths = expanded

    if not log_paths:
        raise SystemExit("No telemetry logs found")

    ndjson_path = Path(args.ndjson)
    dataset_path = Path(args.output)

    total = 0
    # Map session folder -> top_dir (if not provided); assumes telemetry.jsonl is at <session>/telemetry.jsonl and frames are at <session>/frames_top
    session_top_dir_cache: Dict[str, str] = {}

    with dataset_path.open("w", encoding="utf-8") as out, ndjson_path.open("w", encoding="utf-8") as batch_fp:
        for log_path in log_paths:
            entries = sample_entries(load_logs([str(log_path)]), args.sample_sec)
            veh_frame_counter: Dict[int, int] = {}
            inferred_top_dir = args.top_dir or str(log_path.parent / "frames_top")

            for ts, tele in entries:
                top_b64 = ""
                vid = tele.get("veh_id")
                frame_idx = veh_frame_counter.get(vid, 0)

                if args.top_frame:
                    candidate = args.top_frame.format(veh_id=vid, index=frame_idx)
                    top_b64 = maybe_encode_image(candidate)
                elif inferred_top_dir:
                    candidate = find_top_frame(inferred_top_dir, vid, index=frame_idx)
                    top_b64 = maybe_encode_image(candidate)

                if (args.top_dir or args.top_frame) and not top_b64:
                    continue

                messages = build_messages(args.goal, tele, top_b64)
                cid = f"veh-{tele.get('veh_id')}-{int(ts * 1000)}"
                record = {
                    "custom_id": cid,
                    "ts": ts,
                    "veh_id": tele.get("veh_id"),
                    "goal": args.goal,
                    "telemetry": tele,
                    "messages": messages,
                }
                if args.include_image_b64 and top_b64:
                    record["image_b64"] = top_b64
                out.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")

                req = {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": messages,
                        "response_format": {"type": "json_object"},
                    },
                }
                batch_fp.write(json.dumps(req, separators=(",", ":")) + "\n")
                total += 1
                veh_frame_counter[vid] = frame_idx + 1

    print(f"Prepared {total} rows, dataset: {dataset_path}, batch payload: {ndjson_path}")

    # Upload + create batch
    print("Uploading batch NDJSON...")
    file_id = await upload_file(ndjson_path, api_key)
    print(f"File uploaded: {file_id}")

    print("Creating batch job...")
    batch_id = await create_batch(file_id, api_key)
    print(f"Batch created: {batch_id}")

    # Poll
    print("Polling for batch completion...")
    result = await poll_batch(batch_id, api_key)
    status = result.get("status")
    print(f"Batch status: {status}")
    if status != "completed":
        print("Batch did not complete successfully; exiting.")
        return

    # Download output
    batch_output_path = Path(args.batch_output)
    print("Downloading batch output...")
    await download_batch_output(batch_id, api_key, batch_output_path)
    print(f"Saved batch output to {batch_output_path}")

    # Merge
    merged_path = Path(args.merged_output)
    print("Merging responses into dataset...")
    merge_responses(dataset_path, batch_output_path, merged_path)
    print(f"Merged dataset written to {merged_path}")


if __name__ == "__main__":
    asyncio.run(main())
