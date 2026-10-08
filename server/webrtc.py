# server/webrtc.py
"""
WebRTC helpers for CARLA AI-in-the-Loop
---------------------------------------
Provides:
  - CarlaVideoTrack: aiortc MediaStreamTrack that reads the latest JPEG
    from a provided frame_provider(view, veh_id) callable and streams it
    as a video track over WebRTC (VP8/H264).
  - PCS: global set of RTCPeerConnection objects.
  - set_bitrate_cap(pc, max_bitrate): optional max bitrate per sender.
  - close_all_pcs(): gracefully close all active peer connections.

Design notes:
  * This module DOES NOT import server.server to avoid circular imports.
  * server.server imports {CarlaVideoTrack, PCS, set_bitrate_cap, close_all_pcs}.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional, Tuple, Any, Set

import av
import numpy as np
import cv2

from aiortc import RTCPeerConnection, RTCRtpSender
from aiortc.mediastreams import MediaStreamTrack

# Public set of active PeerConnections (shared with server.py)
PCS: Set[RTCPeerConnection] = set()


class CarlaVideoTrack(MediaStreamTrack):
    """
    MediaStreamTrack for CARLA frames fed from a shared JPEG buffer.

    Args:
      frame_provider: Callable(view: str, veh_id: Optional[int]) -> Optional[bytes]
        - Must return the most recent JPEG (bytes) or None if not available.
      view: "front" | "top" | "global"
      veh_id: Optional[int]
      target_fps: float (frames per second)
      scale_to: Optional[Tuple[int,int]] (w,h) — downscale for bandwidth savings
      max_bitrate: Optional[int] (bps). Not enforced here; use set_bitrate_cap.

    Behavior:
      - On each recv(), it attempts to fetch a new JPEG at ~target_fps cadence.
      - If no frame is available, it sends a small black frame (keeps the stream alive).
      - Converts JPEG -> BGR -> (optional resize) -> RGB -> av.VideoFrame.
    """

    kind = "video"

    def __init__(
        self,
        frame_provider: Callable[[str, Optional[int]], Optional[bytes]],
        view: str = "front",
        veh_id: Optional[int] = None,
        target_fps: float = 20.0,
        scale_to: Optional[Tuple[int, int]] = None,
        max_bitrate: Optional[int] = None,  # kept for signature completeness
    ) -> None:
        super().__init__()  # don't forget this
        self._frame_provider = frame_provider
        self._view = view
        self._veh_id = veh_id
        self._period = 1.0 / max(1.0, float(target_fps))
        self._scale_to = scale_to
        self._last_ts = 0.0
        self._clock_base = time.time()

        # cached black fallback
        self._fallback_rgb = self._make_fallback(640, 360)

    # -------------- helpers -----------------

    @staticmethod
    def _jpeg_to_bgr(jpg: bytes) -> Optional[np.ndarray]:
        """Decode JPEG -> BGR ndarray."""
        try:
            arr = np.frombuffer(jpg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img  # BGR or None
        except Exception:
            return None

    @staticmethod
    def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _make_fallback(w: int, h: int) -> np.ndarray:
        """Small black RGB frame with a 'waiting' text."""
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(rgb, "waiting...", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
        return rgb

    def _now_ts(self) -> int:
        """
        Monotonic-ish timestamp in microseconds for WebRTC.
        aiortc expects pts in time_base of 1e6 by default for raw frames.
        """
        return int((time.time() - self._clock_base) * 1_000_000)

    # -------------- MediaStreamTrack API --------------

    async def recv(self) -> av.VideoFrame:
        """
        Called by aiortc pipeline to pull the next frame.
        We rate-limit to _period seconds between frames.
        """
        # maintain target FPS
        now = time.time()
        delta = now - self._last_ts
        if delta < self._period:
            await asyncio.sleep(self._period - delta)
        self._last_ts = time.time()

        # try to get a JPEG from provider
        jpg = None
        try:
            jpg = self._frame_provider(self._view, self._veh_id)
        except Exception:
            jpg = None

        if jpg:
            bgr = self._jpeg_to_bgr(jpg)
        else:
            bgr = None

        if bgr is None:
            rgb = self._fallback_rgb
        else:
            # Optional scale
            if self._scale_to is not None:
                try:
                    w, h = self._scale_to
                    if w > 0 and h > 0:
                        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
                except Exception:
                    pass
            rgb = self._bgr_to_rgb(bgr)

        # Build av.VideoFrame
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = self._now_ts()
        frame.time_base = av.time_base  # microseconds
        return frame


# ---------- Sender bitrate management & cleanup -------------

async def set_bitrate_cap(pc: RTCPeerConnection, max_bitrate_bps: int) -> None:
    """
    Applies a maxBitrate (bps) on all video RTCRtpSenders in the given PeerConnection.
    Note: Browser support varies, but setting RTCRtpEncodingParameters.maxBitrate
    is widely supported for outbound streams.
    """
    try:
        for sender in pc.getSenders():
            if sender.kind != "video":
                continue
            params = sender.getParameters()
            if not params.encodings:
                params.encodings = [{}]
            for enc in params.encodings:
                enc["maxBitrate"] = int(max_bitrate_bps)
            await sender.setParameters(params)
    except Exception:
        # Non-fatal; continue without bitrate cap
        pass


async def close_all_pcs() -> None:
    """Gracefully close and clear all active RTCPeerConnections."""
    to_close = list(PCS)
    for pc in to_close:
        try:
            await pc.close()
        except Exception:
            pass
        try:
            PCS.discard(pc)
        except Exception:
            pass