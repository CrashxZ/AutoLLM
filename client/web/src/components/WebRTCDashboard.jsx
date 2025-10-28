import React, { useEffect, useMemo, useState } from "react";
import useWebSocket from "../hooks/useWebSocket.js";           // your existing WS hook (telemetry + commands)
import useWebRTC from "../hooks/useWebRTC.js";                 // new
import WebRTCVideo from "./WebRTCVideo.jsx";                   // new
import ControlPanel from "./ControlPanelWebRTC.jsx";           // new
import ConfigDrawer from "./ConfigDrawer.jsx";                 // reuse

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const WS_URL   = import.meta.env.VITE_WS_URL   || "ws://localhost:8000/ws_ui";

export default function WebRTCDashboard() {
  // WebSocket: get telemetry + veh ids and send commands
  const { telemetry, vehicles, status, sendCommand } = useWebSocket({ url: WS_URL });
  const connected = status === "open";

  // View + selection
  const [view, setView] = useState("front"); // "front" | "top" | "global"
  const [selectedVehicle, setSelectedVehicle] = useState(null);
  const [showConfig, setShowConfig] = useState(false);
  const [goal, setGoal] = useState("");

  // WebRTC hook for video (front/top/global via aiortc)
  const {
    pcState,
    videoRef,
    connect,
    disconnect,
    renegotiate,
    error: rtcError
  } = useWebRTC({
    apiBase: API_BASE,
    defaultConstraints: {
      fps: 20,
      maxBitrate: 1200000,  // ~1.2 Mbps; tweak to your link
      scaleTo: [960, 540],  // on server side; here informational only
    },
  });

  // auto-select first vehicle when list appears (for front/top)
  useEffect(() => {
    if (view === "global") {
      setSelectedVehicle(null);
      return;
    }
    if (!vehicles?.length) return;
    if (!selectedVehicle || !vehicles.includes(selectedVehicle)) {
      setSelectedVehicle(vehicles[0]);
    }
  }, [vehicles, view]);

  // WebRTC (re)connect when view/veh changes
  useEffect(() => {
    if (view === "global") {
      renegotiate({ view: "global", vehId: null, fps: 20, maxBitrate: 1200000 });
      return;
    }
    if (selectedVehicle) {
      renegotiate({ view, vehId: selectedVehicle, fps: 20, maxBitrate: 1200000 });
    }
  }, [view, selectedVehicle, renegotiate]);

  // Simple REST helpers
  const handleResetSim = async () => {
    try {
      await fetch(`${API_BASE}/reset`, { method: "POST" });
    } finally {
      // clear selections and renegotiate to a blank/global
      setSelectedVehicle(null);
      setView("global");
      renegotiate({ view: "global", vehId: null, fps: 10, maxBitrate: 800000 });
    }
  };

  const handleApplyConfig = async (cfg) => {
    await fetch(`${API_BASE}/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    });
    setShowConfig(false);
  };

  const vehTele = useMemo(() => {
    if (!selectedVehicle) return null;
    return telemetry?.[String(selectedVehicle)] || null;
  }, [telemetry, selectedVehicle]);

  return (
    <div className="h-screen w-screen flex flex-col bg-gray-900 text-gray-100">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold">CARLA WebRTC Dashboard</h1>
          <span className={`px-2 py-1 text-xs rounded ${connected ? "bg-green-600" : "bg-red-600"}`}>
            {connected ? "WS Connected" : "WS Disconnected"}
          </span>
          <span className={`px-2 py-1 text-xs rounded ${pcState === "connected" ? "bg-green-600" : "bg-gray-700"}`}>
            RTC: {pcState}
          </span>
          {rtcError && <span className="text-xs text-red-400">RTC error: {String(rtcError)}</span>}
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => setShowConfig(true)} className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm">
            ⚙️ Configure
          </button>
          <button onClick={handleResetSim} className="px-3 py-1 bg-red-700 hover:bg-red-800 rounded text-sm">
            🔁 Reset
          </button>
          <button onClick={() => connect({ view, vehId: selectedVehicle, fps: 20, maxBitrate: 1200000 })}
                  className="px-3 py-1 bg-indigo-600 hover:bg-indigo-700 rounded text-sm">
            🔌 Connect RTC
          </button>
          <button onClick={disconnect} className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm">
            ⏏️ Disconnect RTC
          </button>
        </div>
      </header>

      {/* Main */}
      <div className="flex flex-1 overflow-hidden">
        {/* Video */}
        <div className="flex-1 relative bg-black flex items-center justify-center">
          <WebRTCVideo ref={videoRef} />
          {/* Overlay HUD */}
          {(vehTele && (view === "front" || view === "top")) && (
            <div className="absolute top-2 left-2 bg-black/60 text-xs rounded p-2 leading-tight">
              <div>ID: {vehTele.veh_id}</div>
              <div>Speed: {vehTele.speed_kmh?.toFixed?.(1)} km/h</div>
              <div>Lane: {vehTele.lane_id}</div>
              <div>LC: {vehTele.lane_change?.state}</div>
            </div>
          )}
          {view === "global" && (
            <div className="absolute top-2 left-2 bg-black/60 text-xs rounded p-2 leading-tight">
              <div>🌍 Global View</div>
            </div>
          )}
        </div>

        {/* Controls */}
        <ControlPanel
          view={view}
          setView={setView}
          vehicles={vehicles}
          selectedVehicle={selectedVehicle}
          onSelectVehicle={(id) => {
            setSelectedVehicle(id);
            // also tell server who is "selected" for ego tx/logic where relevant
            if (id != null) sendCommand({ cmd: "select", veh_id: id });
          }}
          sendCommand={sendCommand}
          goal={goal}
          onGoalChange={setGoal}
          onRenegotiate={() => renegotiate({ view, vehId: selectedVehicle, fps: 20, maxBitrate: 1200000 })}
        />
      </div>

      {/* Config Drawer (no “Apply/Reset” buttons per your request) */}
      {showConfig && (
        <ConfigDrawer
          onClose={() => setShowConfig(false)}
          onApply={handleApplyConfig}
          hideFooterButtons
        />
      )}
    </div>
  );
}