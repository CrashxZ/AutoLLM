"""Record 36 illustrative fleet/front CARLA videos, retaining episode logs."""

from __future__ import annotations

import bisect
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

import camera_recording as gallery

ROOT = gallery.ROOT
V2 = gallery.V2
LABEL = "scenario_gallery_20261008"
VIDEOS = ROOT / "results/videos" / LABEL
DATA = ROOT / "data/experiments" / LABEL
API = "http://127.0.0.1:8000"
SCENARIOS = (
    ("close_reciprocal", "S1_reciprocal_exchange", "S1: Reciprocal exchange"),
    ("contested_merge", "S2_contested_merge", "S2: Contested merge"),
    ("blocked_merge", "S3_blocked_merge", "S3: Blocked merge"),
)
METHODS = (
    ("FCFS_GAP", "FCFS-GAP"),
    ("MAPPO_ADAPTED", "MAPPO-adapted"),
    ("MIND_CAV_DETERMINISTIC", "MIND-CAV-deterministic"),
    ("MIND_CAV_LEARNED", "MIND-CAV-learned"),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


class FleetRecorder(gallery.CameraRecorder):
    def attach(self, vehicle_ids):
        super().attach(vehicle_ids)
        client = gallery.carla.Client("127.0.0.1", 2000)
        client.set_timeout(20)
        self.world = client.get_world()
        self.vehicles = [self.world.get_actor(vid) for vid in vehicle_ids]
        self.sensors[0].stop()
        self.sensors[0].destroy()
        blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
        for key, value in {
            "image_size_x": "1080", "image_size_y": "1080",
            "fov": "60", "sensor_tick": "0.1",
        }.items():
            blueprint.set_attribute(key, value)
        self.sensors[0] = self.world.spawn_actor(blueprint, gallery.carla.Transform())
        self.sensors[0].listen(self._callback("top"))
        self.update_framing()

    def update_framing(self):
        locations = [vehicle.get_location() for vehicle in self.vehicles]
        xs, ys = [p.x for p in locations], [p.y for p in locations]
        half_extent = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 15
        altitude = max(42, half_extent / math.tan(math.radians(30)))
        transform = gallery.carla.Transform(
            gallery.carla.Location(
                x=(min(xs) + max(xs)) / 2, y=(min(ys) + max(ys)) / 2,
                z=max(p.z for p in locations) + altitude,
            ), gallery.carla.Rotation(pitch=-90),
        )
        self.sensors[0].set_transform(transform)
        with (self.frame_dir.parent / "camera_framing.jsonl").open("a") as stream:
            stream.write(json.dumps({
                "sim_frame": self.world.get_snapshot().frame,
                "center_x": transform.location.x, "center_y": transform.location.y,
                "altitude_m": altitude, "fleet_extent_m": 2 * (half_extent - 15),
            }) + "\n")


class FleetClient(gallery.CaptureClient):
    def post(self, url, **kwargs):
        if url.rstrip("/").endswith("/simulation/step"):
            self.recorder.update_framing()
        return super().post(url, **kwargs)


def render(base, video, title, result):
    """Keep simulator timing even when a camera frame is missing."""
    trajectory = [json.loads(line) for line in (base / "run/trajectory.jsonl").read_text().splitlines()]
    first = next(iter(trajectory[0]["vehicles"].values()))["sim_frame"]
    last = next(iter(trajectory[-1]["vehicles"].values()))["sim_frame"]
    images = {}
    for view in ("top", "front"):
        images[view] = {int(p.stem.split("_")[-1]): p for p in (base / "frames" / view).glob("frame_*.jpg")}
        if len(images[view]) < 5:
            raise RuntimeError(f"Insufficient {view} frames")
    start = max(first, *(min(frames) for frames in images.values()))
    end = min(last, *(max(frames) for frames in images.values()))
    targets = list(range(start, end + 1, 2))  # 0.05-s CARLA ticks, 10 fps.
    if len(targets) < 5:
        raise RuntimeError("Insufficient common video duration")
    sequence = base / "encoding_sequence"
    if sequence.exists():
        shutil.rmtree(sequence)
    held = {}
    max_age = {}
    for view, frames in images.items():
        folder = sequence / view
        folder.mkdir(parents=True)
        ids = sorted(frames)
        held[view] = 0
        max_age[view] = 0
        for index, frame in enumerate(targets, 1):
            selected = ids[max(0, bisect.bisect_right(ids, frame) - 1)]
            held[view] += int(frame - selected >= 2)
            max_age[view] = max(max_age[view], (frame - selected) * 0.05)
            os.symlink(frames[selected].resolve(), folder / f"{index:06d}.jpg")
    subtitle = (
        f"{result['completed_command_count']}/{result['planned_command_count']} maneuvers | "
        f"{result['status']} | collision notifications: {result['collision_count']} | "
        "Illustrative rerun; not part of reported results"
    )
    overlay = base / "overlay.png"
    gallery.overlay_image(title, subtitle, overlay)
    video.parent.mkdir(parents=True, exist_ok=True)
    temporary = video.with_name(video.stem + ".partial.mp4")
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-threads", "2",
        "-framerate", "10", "-i", str(sequence / "top/%06d.jpg"),
        "-framerate", "10", "-i", str(sequence / "front/%06d.jpg"),
        "-loop", "1", "-i", str(overlay),
        "-filter_complex_threads", "1", "-filter_complex",
        "[0:v]scale=600:600,pad=600:720:0:60:color=0x101b2b[top];"
        "[1:v]scale=680:383,pad=680:720:0:168:color=0x101b2b[front];"
        "[top][front]hstack=inputs=2[split];[split][2:v]overlay=0:0:shortest=1[out]",
        "-map", "[out]", "-frames:v", str(len(targets)),
        "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    subprocess.run(command, check=True)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(temporary),
    ]))
    stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert int(stream["nb_frames"]) == len(targets)
    assert (int(stream["width"]), int(stream["height"])) == (1280, 720)
    subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-i", str(temporary), "-f", "null", "-"], check=True)
    temporary.replace(video)
    info = {
        "frames": len(targets), "duration_s": float(probe["format"]["duration"]),
        "first_sim_frame": start, "last_sim_frame": end,
        "held_frame_counts": held, "maximum_held_frame_age_s": max_age,
        "original_first_sim_frame": first, "original_last_sim_frame": last,
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
    }
    save_json(base / "video_capture.json", info)
    # Delete only this recording's temporary frames after successful decode.
    shutil.rmtree(sequence)
    shutil.rmtree(base / "frames")
    return info


def update_index(records):
    lines = ["# CARLA scenario videos", "", "One fresh illustrative run per method and scenario–fleet combination. Each video shows a fleet-centred overhead view (left) and vehicle-0 front view (right). Original results are not replaced.", "", "| Scenario | Vehicles | FCFS-GAP | MAPPO adapted | MIND-CAV deterministic | MIND-CAV learned |", "|---|---:|---|---|---|---|"]
    for family, folder, title in SCENARIOS:
        for fleet in (2, 4, 8):
            cells = []
            for method, label in METHODS:
                key = f"{family}:{fleet}:{method}"
                item = records.get(key)
                cells.append(f"[Video]({folder}/{fleet}_vehicles/{label}.mp4)" if item and item.get("status") == "complete" else "Pending")
            lines.append("| " + " | ".join([title, str(fleet)] + cells) + " |")
    lines += ["", "All videos use repetition 4 from the original schedule, selected before recording. Failures and contacts are retained. Temporary camera JPEGs are removed after encoding and full decoding checks; episode trajectories, camera framing, source/model metadata, and MP4s remain available.", "", "Detailed recording outcomes and paths: `manifest.json`. Raw logs: `data/experiments/scenario_gallery_20261008` in the AutoLLM workspace."]
    (VIDEOS / "README.md").write_text("\n".join(lines) + "\n")
    save_json(VIDEOS / "manifest.json", records)


def main():
    global LABEL, VIDEOS, DATA
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="scenario_gallery")
    args = parser.parse_args()
    if not args.label.replace("_", "").isalnum():
        raise ValueError("Use letters, digits and underscores for the label")
    LABEL = args.label
    VIDEOS = ROOT / "results/videos" / LABEL
    DATA = ROOT / "data/experiments" / LABEL
    VIDEOS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    lock = (DATA / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    schedule_path = ROOT / "experiments/historical/carla_v4_schedule.json"
    schedule = json.loads(schedule_path.read_text())
    blocks = {b["block_id"]: b for b in schedule["blocks"]}
    actor_path = V2 / "models/mappo_actor.npz"
    actor = gallery.NumpyActorPolicy.load(str(actor_path))
    manifest_path = VIDEOS / "manifest.json"
    records = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    update_index(records)
    with httpx.Client(timeout=120) as control:
        if control.get(API + "/health").json().get("running"):
            raise RuntimeError("Backend must be idle before recording")
        previous = control.get(API + "/mec/v2/status").json()
        if not previous.get("global_conflict_recovery") or not previous.get("liveness_preparation"):
            raise RuntimeError("Corrected follow-up configuration is required")
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=V2)
        config_path = DATA / "configuration.json"
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (actor_path, V2 / "models/candidate_ranker.npz")}
        diff_hash = hashlib.sha256(diff).hexdigest()
        if config_path.exists():
            existing = json.loads(config_path.read_text())
            if existing["source_diff_sha256"] != diff_hash or existing["model_hashes"] != hashes:
                raise RuntimeError("Source or model changed; use a separate gallery version")
        else:
            save_json(config_path, {
                "created_utc": now(), "claim_status": "illustrative_camera_reruns",
                "cases": 36, "selected_repetition": 4, "api_status": previous,
                "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=V2, text=True).strip(),
                "source_diff_sha256": diff_hash, "model_hashes": hashes,
                "schedule_sha256": hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
                "budget": gallery.RunnerBudget().__dict__,
                "retain_frames": False, "camera_views": ["adaptive_fleet", "front_slot_0"],
            })
            (DATA / "source_diff.patch").write_bytes(diff)
        progress = {"total": 36, "completed": sum(r.get("status") == "complete" for r in records.values()), "status": "running", "updated_utc": now()}
        save_json(DATA / "progress.json", progress)
        try:
            for family, folder, title in SCENARIOS:
                for fleet in (2, 4, 8):
                    block = {**blocks[f"{family}-n{fleet}-r004"], "claim_status": "illustrative_camera_rerun"}
                    for method, label in METHODS:
                        key = f"{family}:{fleet}:{method}"
                        video = VIDEOS / folder / f"{fleet}_vehicles" / f"{label}.mp4"
                        base = DATA / folder / f"{fleet}_vehicles" / method
                        if records.get(key, {}).get("status") == "complete" and video.exists():
                            continue
                        if shutil.disk_usage(ROOT).free < 20 * 1024**3:
                            raise RuntimeError("Less than 20 GiB free")
                        print(f"RECORD {key} seed={block['seed']}", flush=True)
                        base.mkdir(parents=True, exist_ok=True)
                        if (base / "run/metadata.json").exists():
                            result = json.loads((base / "run/metadata.json").read_text())
                        else:
                            if (base / "run").exists():
                                raise RuntimeError(f"Partial episode requires inspection: {base}")
                            recorder = FleetRecorder(base / "frames", 0)
                            try:
                                with FleetClient(recorder, timeout=120) as client:
                                    if method.startswith("MIND_CAV"):
                                        variant = "learned" if method.endswith("LEARNED") else "deterministic"
                                        response = client.post(API + "/mec/v2/ranker", json={"variant": variant})
                                        response.raise_for_status()
                                        assert response.json()["proposer"] == f"constrained-{variant}-ranker"
                                    result = gallery.run_method_episode(
                                        client, API, block, method, base / "run",
                                        actor if method == "MAPPO_ADAPTED" else None,
                                        gallery.RunnerBudget(), archive=False, enable_cameras=False,
                                    )
                            finally:
                                recorder.close()
                        capture = render(base, video, f"{title} | {fleet} vehicles | {label}", result)
                        records[key] = {
                            "status": "complete", "scenario": family, "fleet_size": fleet,
                            "method": method, "seed": block["seed"], "block_id": block["block_id"],
                            "video": str(video), "raw_logs": str(base / "run"),
                            "episode_status": result["status"],
                            "completed_maneuvers": result["completed_command_count"],
                            "planned_maneuvers": result["planned_command_count"],
                            "collision_notifications": result["collision_count"],
                            "elapsed_sim_s": result["elapsed_sim_s"], **capture,
                        }
                        update_index(records)
                        progress.update(completed=sum(r.get("status") == "complete" for r in records.values()), last_case=key, updated_utc=now(), free_gib=round(shutil.disk_usage(ROOT).free / 1024**3, 2))
                        save_json(DATA / "progress.json", progress)
                        print(f"DONE {key}: {result['status']}, {capture['duration_s']:.1f}s video", flush=True)
            progress.update(status="complete", updated_utc=now())
        except Exception as error:
            progress.update(status="stopped_with_error", error=str(error), updated_utc=now())
            raise
        finally:
            save_json(DATA / "progress.json", progress)
            variant = "learned" if previous.get("ranker_model_loaded") else "deterministic"
            control.post(API + "/mec/v2/ranker", json={"variant": variant}).raise_for_status()


if __name__ == "__main__":
    main()
