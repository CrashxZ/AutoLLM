// client/web/src/components/Dashboard.jsx
import React, { useEffect, useMemo, useRef, useState, useCallback } from "react";
import ControlPanel from "./ControlPanel.jsx";
import IntentPanel from "./IntentPanel.jsx";
import ConfigDrawer from "./ConfigDrawer.jsx";
import IntentGeneratorPanel from "./IntentGeneratorPanel.jsx";

import useWebSocket from "../hooks/useWebSocket.js";       // default export (your hook)
import useFrameCapture from "../hooks/useFrameCapture.js"; // default export (legacy capture)

/** Environment + API endpoints */
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const WS_URL   = import.meta.env.VITE_WS_URL   || "ws://localhost:8000/ws_ui";

/** Valid camera views for the main display */
const VIEWS = ["front", "top", "global"];

/**
 * Dashboard (MJPEG version)
 * - Uses /video, /video_top, /video_global for smooth streaming (<img src=...> multipart/mjpeg)
 * - Keeps legacy frame capture via useFrameCapture (uses /frame/{veh}.jpg internally)
 * - Preserves your config/reset/export/intent controls and selectors
 */

function IntentionDivider() {
  return (
    <div className="w-full bg-gray-800 text-center py-1 border-t border-b border-gray-700 text-xs text-gray-400">
      Vehicular AI Intention Generator
    </div>
  );
}

export default function Dashboard() {
  // ---- WebSocket / Telemetry ----
  const { telemetry, vehicles, status, sendCommand } = useWebSocket({ url: WS_URL });
  const connected = status === "open";

  // ---- Frame Capture (legacy) ----
  const { isCapturing, startCapture, stopCapture, exportZip } = useFrameCapture({
    fetchFrameUrl: (vehId) => `${API_BASE}/frame/${vehId}.jpg`, // legacy single-frame fetch
    telemetrySource: telemetry,
  });

  // ---- UI State ----
  const [selectedVehicle, setSelectedVehicle] = useState(null);
  const [mode, setMode] = useState(localStorage.getItem("control_mode") || "USER");
  const [goal, setGoal] = useState("");
  const [showConfig, setShowConfig] = useState(false);
  const [showIntent, setShowIntent] = useState(false);
  const [view, setView] = useState("front"); // "front" | "top" | "global"

  // ---- FPS tracker for MJPEG <img> ----
  const [fps, setFps] = useState(0);
  const lastLoadRef = useRef(Date.now());
  useEffect(() => {
    const handle = setInterval(() => {
      const now = Date.now();
      const dt = now - lastLoadRef.current; // ms per frame
      if (dt > 0 && dt < 2000) setFps(1000 / dt);
    }, 1000);
    return () => clearInterval(handle);
  }, []);

  // Auto-select first vehicle when list changes; clear if no vehicles
  useEffect(() => {
    if (!vehicles || vehicles.length === 0) {
      setSelectedVehicle(null);
      // If you're capturing, stop on empty set (safety)
      if (isCapturing) stopCapture();
      return;
    }
    if (selectedVehicle == null || !vehicles.includes(selectedVehicle)) {
      setSelectedVehicle(vehicles[0]);
    }
  }, [vehicles]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
  if (!vehicles?.length) setSelectedVehicle(null);
  else if (selectedVehicle == null || !vehicles.includes(selectedVehicle)) {
    setSelectedVehicle(vehicles[0]);
  }
  }, [vehicles]);

  // Derived telemetry for selected vehicle
  const vehTele = useMemo(() => {
    if (!selectedVehicle) return null;
    return telemetry?.[String(selectedVehicle)] || null;
  }, [telemetry, selectedVehicle]);

  // Command wrapper to always include veh_id when needed
  const sendWrapped = useCallback(
    (cmd) => {
      if (view === "global") {
        // Some commands don't need a veh_id; but user actions likely target a vehicle.
        // We still include veh_id when available.
      }
      if (selectedVehicle) {
        sendCommand({ ...cmd, veh_id: cmd.veh_id ?? selectedVehicle });
      } else {
        // Allow mode changes or global actions without veh_id
        sendCommand(cmd);
      }
    },
    [selectedVehicle, view, sendCommand]
  );

  // Build MJPEG stream URL by view
  const streamUrl = useMemo(() => {
    if (view === "global") {
      return `${API_BASE}/video_global`;
    }
    if (!selectedVehicle) return "";
    if (view === "front") return `${API_BASE}/video/${selectedVehicle}`;
    return `${API_BASE}/video_top/${selectedVehicle}`; // "top"
  }, [view, selectedVehicle]);

  // Main frame <img> load handler -> update fps
  const onFrameLoad = () => {
    lastLoadRef.current = Date.now();
  };

  // ---- Actions ----
  const handleGoalSubmit = (text) => setGoal(text);

  const handleResetSim = async () => {
    try {
      await fetch(`${API_BASE}/reset`, { method: "POST" });
      // Local UI cleanup (server also broadcasts empty veh list)
      setSelectedVehicle(null);
    } catch (e) {
      console.error("Reset failed:", e);
    }
  };

  const handleApplyConfig = async (cfg) => {
    try {
      await fetch(`${API_BASE}/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg),
      });
      setShowConfig(false);
    } catch (e) {
      console.error("Config failed:", e);
    }
  };

  // Safe capture start per current vehicle
  const handleStartCapture = () => {
    if (!selectedVehicle) {
      alert("Select a vehicle to start capture.");
      return;
    }
    startCapture(selectedVehicle);
  };

  // ---- UI ----
  return (
    <div className="h-screen w-screen flex flex-col bg-gray-900 text-white">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold">CARLA AI-in-the-Loop Dashboard (MJPEG)</h1>
          <span
            className={`px-2 py-1 text-xs rounded ${
              connected ? "bg-green-600" : "bg-red-600"
            }`}
          >
            {connected ? "Connected" : "Disconnected"}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowConfig(true)}
            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
          >
            ⚙️ Configure
          </button>
          <button
            onClick={handleResetSim}
            className="px-3 py-1 bg-red-700 hover:bg-red-800 rounded text-sm"
          >
            🔁 Reset
          </button>
          <button
            onClick={exportZip}
            className="px-3 py-1 bg-green-700 hover:bg-green-800 rounded text-sm"
          >
            💾 Export Logs
          </button>
        </div>
      </header>

      {/* Main */}
      <div className="flex flex-1 overflow-hidden">
        {/* Video area */}
        <div className="flex-1 relative bg-black flex flex-col">
          {/* View selector */}
          <div className="p-2 flex items-center gap-2 bg-gray-800 border-b border-gray-700">
            <span className="text-xs text-gray-300">View:</span>
            <div className="inline-flex rounded overflow-hidden border border-gray-600">
              {VIEWS.map((v) => (
                <button
                  key={v}
                  onClick={() => setView(v)}
                  className={`px-3 py-1 text-sm ${
                    view === v ? "bg-gray-700" : "bg-gray-800 hover:bg-gray-700"
                  }`}
                >
                  {v === "front" ? "Front" : v === "top" ? "Top" : "Global"}
                </button>
              ))}
            </div>
            <div className="ml-auto text-xs text-gray-400">
              FPS: {fps.toFixed(1)}
            </div>
          </div>
          

          {/* Stream */}
          <div className="flex-1 min-h-0 flex items-center justify-center">
            {streamUrl ? (
              <img
                src={streamUrl}
                alt={`CARLA ${view} stream`}
                onLoad={onFrameLoad}
                className="object-contain w-full h-full select-none"
                crossOrigin="anonymous"
              />
            ) : (
              <div className="text-gray-400">
                {view === "global"
                  ? "Global stream is initializing…"
                  : "Select a vehicle to view stream."}
              </div>
            )}
          </div>

          {/* Overlay HUD (only shows veh telemetry for front/top) */}
          {(view === "front" || view === "top") && vehTele && (
            <div className="absolute top-2 left-2 bg-black/60 text-xs rounded p-2 leading-tight">
              <div>ID: {vehTele.veh_id ?? selectedVehicle}</div>
              {Number.isFinite(vehTele.speed_kmh) && (
                <div>Speed: {vehTele.speed_kmh.toFixed?.(1)} km/h</div>
              )}
              <div>Lane: {vehTele.lane_id}</div>
              <div>LC: {vehTele.lane_change?.state} {vehTele.lane_change?.direction}</div>
              <div>Conn: {status}</div>
              <div>Mode: {mode}</div>
            </div>
          )}
        </div>

        {/* Controls (vehicle select, mode, goal, manual/LLM, capture & intent) */}
        <ControlPanel
          view={view}
          onViewChange={(v) => setView(v)}
          mode={mode}
          onModeChange={setMode}
          selectedVehicle={selectedVehicle}
          setSelectedVehicle={(id) => {
            setSelectedVehicle(id);
            if (view === "global") setView("front"); // if user selects a vehicle while on global, flip to a vehicle view
          }}
          vehicles={vehicles}
          sendCommand={sendWrapped}
          goal={goal}
          onGoalSubmit={handleGoalSubmit}
          startCapture={handleStartCapture}
          stopCapture={stopCapture}
          isCapturing={isCapturing}
          openIntentPanel={() => setShowIntent(true)}
        />
      </div>
      {/* Vehicular AI panel (intent only) */}
      <IntentionDivider />
      <IntentGeneratorPanel
        vehicles={vehicles}
        telemetry={telemetry}
        goal={goal}
        periodSec={10}
      />
      {/* Intent Drawer */}
      {showIntent && (
        <div className="fixed right-0 top-0 bottom-0 w-80 border-l border-gray-700 bg-gray-900 z-50">
          <IntentPanel
            sendCommand={sendCommand}
            selectedVehId={selectedVehicle}
          />
          <button
            onClick={() => setShowIntent(false)}
            className="absolute top-2 right-2 text-gray-400 hover:text-white text-lg"
          >
            ✖
          </button>
        </div>
      )}

      {/* Config Drawer */}
      {showConfig && (
        <ConfigDrawer
          onClose={() => setShowConfig(false)}
          onApply={handleApplyConfig}
        />
      )}

      {/* Footer */}
      <footer className="text-center py-1 text-xs bg-gray-800 border-t border-gray-700">
        © {new Date().getFullYear()} CPS Lab, University of Connecticut
      </footer>
    </div>
  );
}