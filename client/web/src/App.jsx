// client/web/src/App.jsx
/**
 * Main CARLA AI-in-the-Loop Dashboard
 * ------------------------------------
 * React + Tailwind (Vite) entry point.
 *
 * Responsibilities:
 *  - Layout video feed, control panels, telemetry logs
 *  - Manage mode (USER / LLM Local / LLM API)
 *  - Integrate WebSocket + MJPEG stream + config/reset controls
 *  - Provide drawers for configuration and intent
 *
 * Assumes supporting components exist:
 *  - components/VideoFeed.jsx
 *  - components/ControlPanel.jsx
 *  - components/TelemetryPanel.jsx
 *  - components/ConfigDrawer.jsx
 *  - components/IntentPanel.jsx
 *  - hooks/useWebSocket.js
 *  - hooks/useFrameCapture.js
 */

import { useState, useEffect } from "react";
import VideoFeed from "./components/VideoFeed.jsx";
import ControlPanel from "./components/ControlPanel.jsx";
import TelemetryPanel from "./components/TelemetryPanel.jsx";
import ConfigDrawer from "./components/ConfigDrawer.jsx";
import IntentPanel from "./components/IntentPanel.jsx";
import useWebSocket from "./hooks/useWebSocket.js";
import useFrameCapture from "./hooks/useFrameCapture.js";

export default function App() {
  const [connected, setConnected] = useState(false);
  const [mode, setMode] = useState("USER"); // USER | LLM_LOCAL | LLM_API
  const [goal, setGoal] = useState("");
  const [selectedVehicle, setSelectedVehicle] = useState(null);
  const [showConfig, setShowConfig] = useState(false);
  const [showIntent, setShowIntent] = useState(false);

  // --- WebSocket telemetry ---
  const { telemetry, sendCommand, connect, disconnect, vehicles, status } =
    useWebSocket({
      url: "ws://localhost:8000/ws_ui",
      onOpen: () => setConnected(true),
      onClose: () => setConnected(false),
    });

  // --- Frame capture + logs ---
  const { startCapture, stopCapture, exportZip, isCapturing } = useFrameCapture({
    fetchFrameUrl: (vehId) => `http://localhost:8000/frame/top/${vehId}.jpg`,
    telemetrySource: telemetry,
  });

  // Auto-select first vehicle when available
  useEffect(() => {
    if (vehicles.length && !selectedVehicle) setSelectedVehicle(vehicles[0]);
  }, [vehicles]);

  // --- Handlers ---
  const handleModeChange = (newMode) => setMode(newMode);
  const handleGoalSubmit = (g) => setGoal(g);
  const handleSendCommand = (cmd) => {
    if (selectedVehicle) sendCommand({ ...cmd, veh_id: selectedVehicle });
  };
  const handleReset = async () => {
    await fetch("http://localhost:8000/reset", { method: "POST" });
  };
  const handleConfigApply = async (cfg) => {
    await fetch("http://localhost:8000/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    });
    setShowConfig(false);
  };

  return (
    <div className="w-full h-screen flex flex-col bg-gray-900 text-gray-100 overflow-hidden">
      {/* Header Toolbar */}
      <header className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-4">
          <h1 className="text-xl font-semibold tracking-wide">
            🚗 CARLA AI-in-the-Loop Dashboard
          </h1>
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
            className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
          >
            ⚙️ Configure
          </button>
          <button
            onClick={handleReset}
            className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-sm"
          >
            🔄 Reset Simulation
          </button>
          <button
            onClick={() =>
              connected ? disconnect() : connect()
            }
            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
          >
            {connected ? "Disconnect" : "Connect"}
          </button>
          <button
            onClick={exportZip}
            className="px-3 py-1 bg-purple-700 hover:bg-purple-800 rounded text-sm"
          >
            ⬇️ Export Logs
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left / Center: Video Feed */}
        <div className="flex-1 flex flex-col overflow-hidden">
          // in App.jsx
          const [view, setView] = useState("front");

          <VideoFeed
            view={view}
            selectedVehicle={selectedVehicle}
            telemetry={telemetry}
            connected={connected}
          />

// in App.jsx
const [view, setView] = useState("front");

<VideoFeed
  view={view}
  selectedVehicle={selectedVehicle}
  telemetry={telemetry}
  connected={connected}
/>
          <TelemetryPanel telemetry={telemetry} />
        </div>

        {/* Right Sidebar: Controls */}
        <ControlPanel
        view={view}
        setView={setView}
        mode={mode}
        onModeChange={handleModeChange}
        selectedVehicle={selectedVehicle}
        setSelectedVehicle={setSelectedVehicle}
        vehicles={vehicles}
        sendCommand={handleSendCommand}
        goal={goal}
        onGoalSubmit={handleGoalSubmit}
        startCapture={startCapture}
        stopCapture={stopCapture}
        isCapturing={isCapturing}
        openIntentPanel={() => setShowIntent(true)}
        />
      </div>

      {/* Drawers */}
      {showConfig && (
        <ConfigDrawer
          onClose={() => setShowConfig(false)}
          onApply={handleConfigApply}
        />
      )}
      {showIntent && (
        <IntentPanel
          onClose={() => setShowIntent(false)}
          mode={mode}
          goal={goal}
          telemetry={telemetry}
          sendCommand={sendCommand}
        />
      )}

      {/* Footer */}
      <footer className="text-center py-1 text-xs bg-gray-800 border-t border-gray-700">
        © {new Date().getFullYear()} CPS Lab, University of Connecticut
      </footer>
    </div>
  );
}
