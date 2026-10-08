#!/usr/bin/env python3
"""Record camera reruns for the v4 CARLA methods and scenario families.

Every case is a new illustrative rerun. Original campaign data are read-only.
Only one vehicle receives extra recording cameras to keep I/O manageable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(
    0,
    os.environ.get("CARLA_PYTHON_EGG", ""),
)

import carla
import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
V2 = Path(__file__).resolve().parents[1]
CAMPAIGN = V2 / "data/experiments/carla_conflict_active_confirmatory_v4_20260916"
DATA_OUT = ROOT / "data/experiments/carla_camera_gallery_20260924"
VIDEO_OUT = ROOT / "results/videos/mindcav_explainer/gallery"
METHODS = ("FCFS_GAP", "MAPPO_ADAPTED", "MIND_CAV_DETERMINISTIC", "MIND_CAV_LEARNED")
FAMILIES = ("close_reciprocal", "contested_merge", "blocked_merge")
sys.path.insert(0, str(V2))

from marl.numpy_policy import NumpyActorPolicy  # noqa: E402
from scripts.run_carla_paired_coordination import RunnerBudget, run_method_episode  # noqa: E402






class CameraRecorder:
    def __init__(self, frame_dir: Path, slot: int) -> None:
        self.frame_dir = frame_dir
        self.slot = slot
        self.sensors: list[carla.Sensor] = []
        self.vehicle_id: int | None = None
        for view in ("top", "front"):
            (frame_dir / view).mkdir(parents=True, exist_ok=True)

    def attach(self, vehicle_ids: list[int]) -> None:
        self.vehicle_id = int(vehicle_ids[self.slot])
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(20.0)
        world = client.get_world()
        actor = world.get_actor(self.vehicle_id)
        if actor is None:
            raise RuntimeError(f"CARLA vehicle {self.vehicle_id} missing after config")
        for view, resolution, transform in (
            (
                "top", (720, 720), carla.Transform(carla.Location(z=55.0), carla.Rotation(pitch=-90.0))),
            (
                "front", (960, 540), carla.Transform(carla.Location(x=0.8, z=1.4), carla.Rotation())),
        ):
            blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
            blueprint.set_attribute("image_size_x", str(resolution[0]))
            blueprint.set_attribute("image_size_y", str(resolution[1]))
            blueprint.set_attribute("fov", "90.0" if view == "front" else "60.0")
            blueprint.set_attribute("sensor_tick", "0.1")
            sensor = world.spawn_actor(blueprint, transform, attach_to=actor)
            sensor.listen(self._callback(view))
            self.sensors.append(sensor)

    def _callback(self, view: str):
        target = self.frame_dir / view

        def save(image: carla.Image) -> None:
            pixels = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
            bgr = cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
            success, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if success:
                (target / f"frame_{image.frame:08d}.jpg").write_bytes(encoded.tobytes())

        return save

    def close(self) -> None:
        for sensor in self.sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
        self.sensors.clear()


class CaptureClient(httpx.Client):
    def __init__(self, recorder: CameraRecorder, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.recorder = recorder

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        response = super().post(url, **kwargs)
        if url.rstrip("/").endswith("/config") and response.is_success:
            self.recorder.attach([int(value) for value in response.json()["veh_ids"]])
        return response


def overlay_image(label: str, subtitle: str, target: Path) -> None:
    image = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 16, 1240, 88), radius=13, fill=(10, 22, 37, 220))
    draw.rounded_rectangle((18, 657, 1240, 702), radius=11, fill=(10, 22, 37, 220))
    bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 27)
    regular = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    draw.text((37, 31), label, font=bold, fill=(240, 249, 255, 255))
    draw.text((38, 669), subtitle, font=regular, fill=(191, 216, 230, 255))
    image.save(target)
