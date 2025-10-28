// client/web/src/hooks/useFrameCapture.js
/**
 * useFrameCapture
 * ----------------
 * Client-side frame capture + dataset export (ZIP).
 *
 * Responsibilities:
 *  - Pulls JPEG frames from server (e.g., /frame/{veh_id}.jpg) at a fixed rate
 *  - Pairs each frame with latest telemetry for that veh_id
 *  - Accumulates entries into in-memory JSONL + CSV buffers
 *  - Exports a ZIP containing images/ + telemetry.jsonl + telemetry.csv
 *
 * Usage:
 *   const { startCapture, stopCapture, exportZip, isCapturing } = useFrameCapture({
 *     fetchFrameUrl: (vehId) => `http://localhost:8000/frame/${vehId}.jpg`,
 *     telemetrySource: telemetryObj, // from useWebSocket
 *     defaultVehId: null,            // optional
 *     filenamePrefix: "dataset",     // optional
 *   });
 *
 *   // Start for selected vehicle:
 *   startCapture(selectedVehicleId);
 *   // ... later:
 *   stopCapture();
 *   // Export:
 *   await exportZip();
 */

import { useCallback, useEffect, useRef, useState } from "react";
import JSZip from "jszip";
import { saveAs } from "file-saver";

export default function useFrameCapture({
  fetchFrameUrl,
  telemetrySource,
  defaultVehId = null,
  filenamePrefix = "carla_capture",
} = {}) {
  const [isCapturing, setIsCapturing] = useState(false);
  const [activeVehId, setActiveVehId] = useState(defaultVehId);
  const [count, setCount] = useState(0);

  // Buffers
  const jsonlRef = useRef([]);   // each item is already a JSON string with trailing \n
  const csvRef = useRef([]);     // CSV rows as strings (header in slot 0)
  const imagesRef = useRef([]);  // { filename, blob }

  // timers
  const timerRef = useRef(null);

  // local config
  const getHz = () => {
    const s = Number(localStorage.getItem("snapshot_rate_hz") || 2);
    return (Number.isFinite(s) && s > 0 && s <= 30) ? s : 2;
  };

  const controlMode = () => {
    // Stored by ControlPanel/Config in localStorage (USER | LLM_LOCAL | LLM_API)
    return localStorage.getItem("control_mode") || "USER";
  };

  // CSV header initialization
  useEffect(() => {
    if (!csvRef.current.length) {
      csvRef.current.push([
        "ts",
        "frame_id",
        "veh_id",
        "control_mode",
        "user_cmd",
        "llm_cmd",
        "pose_x",
        "pose_y",
        "pose_z",
        "yaw",
        "vel_x",
        "vel_y",
        "vel_z",
        "speed_kmh",
        "lane_id",
        "lane_type",
        "is_junction",
        "distance_to_center",
        "goal_distance",
        "lane_change_state",
        "lane_change_dir",
        "frame_path",
      ].join(","));
    }
  }, []);

  // Try to infer a vehicle id if not set when capturing
  const inferVehId = useCallback(() => {
    if (!telemetrySource) return null;
    const keys = Object.keys(telemetrySource);
    if (!keys.length) return null;
    // Prefer first numeric-sorted by veh_id
    const entries = keys
      .map((k) => telemetrySource[k])
      .filter(Boolean)
      .sort((a, b) => (a.veh_id ?? 0) - (b.veh_id ?? 0));
    return entries.length ? entries[0].veh_id : null;
  }, [telemetrySource]);

  const takeSnapshot = useCallback(async () => {
    const vehId = activeVehId ?? inferVehId();
    if (!vehId) return; // can't snapshot without a vehicle id

    const url = fetchFrameUrl?.(vehId);
    if (!url) return;

    // Resolve telemetry for this veh
    const tele = getTelemetryForVeh(telemetrySource, vehId);

    try {
      const resp = await fetch(url, { cache: "no-cache" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const blob = await resp.blob();

      const frameId = count; // local monotonic counter (server sim_frame not exposed here)
      const ts = tele?.ts ?? Date.now() / 1000;
      const fname = `images/frame_${String(frameId).padStart(6, "0")}.jpg`;

      // JSONL item
      const jsonItem = {
        ts,
        frame_id: frameId,
        veh_id: vehId,
        control_mode: controlMode(),
        user_cmd: null,         // client doesn't track per-frame user_cmd by default
        llm_cmd: null,          // optionally fill via UI plumbing later
        pose: safePick(tele, "pose", { x: null, y: null, z: null, yaw: null }),
        vel: safePick(tele, "vel", { x: null, y: null, z: null }),
        speed_kmh: tele?.speed_kmh ?? null,
        lane_id: tele?.lane_id ?? null,
        lane_type: tele?.lane_type ?? null,
        is_junction: tele?.is_junction ?? null,
        distance_to_center: tele?.distance_to_center ?? null,
        goal_distance: tele?.goal_distance ?? null,
        lane_change_state: tele?.lane_change?.state ?? null,
        lane_change_direction: tele?.lane_change?.direction ?? null,
        frame_path: fname,
      };
      jsonlRef.current.push(JSON.stringify(jsonItem) + "\n");

      // CSV row
      csvRef.current.push([
        fmt(ts, 3),
        frameId,
        vehId,
        controlMode(),
        "", "", // user_cmd, llm_cmd
        n(tele?.pose?.x), n(tele?.pose?.y), n(tele?.pose?.z), n(tele?.pose?.yaw),
        n(tele?.vel?.x), n(tele?.vel?.y), n(tele?.vel?.z),
        n(tele?.speed_kmh),
        v(tele?.lane_id),
        s(tele?.lane_type),
        b(tele?.is_junction),
        n(tele?.distance_to_center),
        n(tele?.goal_distance),
        s(tele?.lane_change?.state),
        s(tele?.lane_change?.direction),
        fname,
      ].join(","));

      // Image
      imagesRef.current.push({ filename: fname, blob });

      // bump the local frame counter
      setCount((c) => c + 1);
    } catch (e) {
      // Non-fatal; keep trying on next tick
      // console.warn("[capture] snapshot failed:", e);
    }
  }, [activeVehId, inferVehId, fetchFrameUrl, telemetrySource, count]);

  const startCapture = useCallback((vehId) => {
    if (isCapturing) return;
    const id = vehId ?? inferVehId();
    if (!id) {
      alert("No vehicle available to capture. Select a vehicle first.");
      return;
    }
    setActiveVehId(id);
    setIsCapturing(true);
    setCount(0);
    jsonlRef.current = [];
    // Keep CSV header; reset to only header row
    csvRef.current = [csvRef.current[0]];
    imagesRef.current = [];

    const hz = getHz();
    const periodMs = Math.max(20, Math.floor(1000 / hz)); // limit to >= 50 FPS theoretical min
    // immediate first shot
    takeSnapshot();
    // schedule subsequent shots
    timerRef.current = setInterval(takeSnapshot, periodMs);
  }, [isCapturing, inferVehId, takeSnapshot]);

  const stopCapture = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsCapturing(false);
  }, []);

  const exportZip = useCallback(async () => {
    // Build a ZIP with /images, telemetry.jsonl, telemetry.csv, and meta.json
    const zip = new JSZip();

    // images
    const folder = zip.folder("images");
    for (const { filename, blob } of imagesRef.current) {
      const short = filename.startsWith("images/") ? filename.slice(7) : filename;
      folder.file(short, blob);
    }

    // JSONL
    const jsonlBlob = new Blob(jsonlRef.current, { type: "application/x-ndjson" });
    zip.file("telemetry.jsonl", jsonlBlob);

    // CSV
    const csvBlob = new Blob([csvRef.current.join("\n") + "\n"], { type: "text/csv" });
    zip.file("telemetry.csv", csvBlob);

    // Meta
    const meta = {
      created_at: new Date().toISOString(),
      veh_id: activeVehId,
      frames: imagesRef.current.length,
      control_mode: controlMode(),
      snapshot_rate_hz: getHz(),
      schema: {
        jsonl: "ts, frame_id, veh_id, control_mode, user_cmd, llm_cmd, pose{x,y,z,yaw}, vel{x,y,z}, speed_kmh, lane_id, lane_type, is_junction, distance_to_center, goal_distance, lane_change_state, lane_change_direction, frame_path",
        csv: csvRef.current[0],
      },
    };
    zip.file("meta.json", JSON.stringify(meta, null, 2));

    const content = await zip.generateAsync({ type: "blob" });
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    saveAs(content, `${filenamePrefix}_${stamp}.zip`);
  }, [activeVehId]);

  // Safety: stop timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  return {
    isCapturing,
    startCapture,
    stopCapture,
    exportZip,
    frameCount: count,
    activeVehId,
  };
}

/* ---------------- helpers ---------------- */

function getTelemetryForVeh(telemetry, vehId) {
  if (!telemetry) return null;
  const key = String(vehId);
  return telemetry[key] || telemetry[vehId] || null;
}

function safePick(obj, key, fallback) {
  if (!obj || typeof obj !== "object") return fallback;
  const v = obj[key];
  return v == null ? fallback : v;
}

// numeric
function n(x) {
  return (x == null || Number.isNaN(Number(x))) ? "" : String(Number(x));
}
// string
function s(x) {
  if (x == null) return "";
  const t = String(x);
  // remove commas and newlines for CSV hygiene
  return t.replace(/[\r\n,]+/g, " ").trim();
}
// bool to 0/1
function b(x) {
  return x ? "1" : "0";
}
function v(x) {
  // lane_id may be negative in CARLA (left/right); keep raw numeric if present
  return (x == null) ? "" : String(x);
}
function fmt(x, d = 3) {
  const n = Number(x);
  if (!Number.isFinite(n)) return "";
  return n.toFixed(d);
}
