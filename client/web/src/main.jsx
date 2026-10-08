// client/web/src/main.jsx
/**
 * main.jsx
 * ----------
 * Entry point for CARLA AI-in-the-loop Web Client (Vite + React + Tailwind).
 *
 * Responsibilities:
 *  - Initialize WebSocket + FrameCapture hooks
 *  - Manage global state: selected vehicle, mode, goal, intent panel
 *  - Display main layout: video + telemetry + side panels
 *  - Provide buttons for configuration drawer + reset simulation
 */

import React, { useState } from "react";
import ReactDOM from "react-dom/client";
import "./index.css";

import useWebSocket from "./hooks/useWebSocket";
import useFrameCapture from "./hooks/useFrameCapture";

// import ControlPanel from "./components/ControlPanel";
// import IntentPanel from "./components/IntentPanel";
// import ConfigDrawer from "./components/ConfigDrawer";
import Dashboard from "./components/Dashboard";
import WebRTCDashboard from "./components/WebRTCDashboard";

const API_BASE = "http://localhost:8000";

function App() {
  // --- WebSocket / Telemetry ---
  const { telemetry, vehicles, status, sendCommand } = useWebSocket({
    url: "ws://localhost:8000/ws_ui",
  });

  // --- Frame Capture ---
  const {
    isCapturing,
    startCapture,
    stopCapture,
    exportZip,
  } = useFrameCapture({
    fetchFrameUrl: (vehId) => `${API_BASE}/frame/${vehId}.jpg`,
    telemetrySource: telemetry,
  });

  // --- UI State ---
  const [selectedVehicle, setSelectedVehicle] = useState(null);
  const [mode, setMode] = useState(localStorage.getItem("control_mode") || "USER");
  const [goal, setGoal] = useState("");
  const [showConfig, setShowConfig] = useState(false);
  const [showIntent, setShowIntent] = useState(false);

  // --- Handlers ---
  const handleGoalSubmit = (text) => {
    setGoal(text);
  };

  const handleApplyConfig = async (cfg) => {
    await fetch(`${API_BASE}/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    });
  };

  const handleResetSim = async () => {
    await fetch(`${API_BASE}/reset`, { method: "POST" });
  };

  // --- Derived telemetry for selected veh ---
  const veh = selectedVehicle ? telemetry[selectedVehicle] : null;

  return (
    <div className="h-screen w-screen flex flex-col bg-gray-900 text-white">
      {/* Top Bar */}
      <header className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <h1 className="text-lg font-semibold">CARLA AI-in-the-Loop Dashboard</h1>
        <div className="flex gap-2 items-center">
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
            🔁 Reset Simulation
          </button>
          <button
            onClick={exportZip}
            className="px-3 py-1 bg-green-700 hover:bg-green-800 rounded text-sm"
          >
            💾 Export Logs
          </button>
        </div>
      </header>

      {/* Main Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Video + Telemetry */}
        <div className="flex-1 relative bg-black flex items-center justify-center">
          {selectedVehicle ? (
            <img
              src={`${API_BASE}/frame/${selectedVehicle}.jpg?${Date.now()}`}
              alt="CARLA Stream"
              className="object-contain max-h-full max-w-full"
            />
          ) : (
            <div className="text-gray-400">Select a vehicle to view stream.</div>
          )}

          {/* Telemetry overlay (simple) */}
          {veh && (
            <div className="absolute top-2 left-2 bg-black/60 text-xs rounded p-2 leading-tight">
              <div>ID: {veh.veh_id}</div>
              <div>Speed: {veh.speed_kmh?.toFixed?.(1)} km/h</div>
              <div>Lane: {veh.lane_id}</div>
              <div>LC State: {veh.lane_change?.state}</div>
              <div>Mode: {mode}</div>
              <div>Conn: {status}</div>
            </div>
          )}
        </div>

        {/* Control Panel */}
        <ControlPanel
          mode={mode}
          onModeChange={setMode}
          selectedVehicle={selectedVehicle}
          setSelectedVehicle={setSelectedVehicle}
          vehicles={vehicles}
          sendCommand={sendCommand}
          goal={goal}
          onGoalSubmit={handleGoalSubmit}
          startCapture={startCapture}
          stopCapture={stopCapture}
          isCapturing={isCapturing}
          openIntentPanel={() => setShowIntent(true)}
        />

        {/* Intent Panel */}
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
      </div>

      {/* Config Drawer */}
      {showConfig && (
        <ConfigDrawer
          onClose={() => setShowConfig(false)}
          onApply={handleApplyConfig}
        />
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Dashboard />
  </React.StrictMode>
);
