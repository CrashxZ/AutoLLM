import React, { useEffect, useState } from "react";

/**
 * ControlPanel for WebRTC dashboard
 * - View buttons (front/top/global)
 * - Vehicle selector (disabled in global)
 * - Manual controls (lane left/right, speed up/down, brake, release)
 * - Goal text (for later AI panel; just stored in parent)
 *
 * Props:
 *   view, setView
 *   vehicles (number[])
 *   selectedVehicle, onSelectVehicle(id)
 *   sendCommand(payload)
 *   goal, onGoalChange(text)
 *   onRenegotiate()  // to refresh RTC when needed
 */
export default function ControlPanel({
  view, setView,
  vehicles,
  selectedVehicle, onSelectVehicle,
  sendCommand,
  goal, onGoalChange,
  onRenegotiate,
}) {
  const [goalInput, setGoalInput] = useState(goal || "");

  useEffect(() => { setGoalInput(goal || ""); }, [goal]);

  const pickVeh = (e) => {
    const id = e.target.value ? Number(e.target.value) : null;
    onSelectVehicle(id);
    if (id != null) {
      // tell server for ego-related logic + base64 ego in /ws_ui if you use it
      sendCommand({ cmd: "select", veh_id: id });
    }
    onRenegotiate?.();
  };

  const clickView = (v) => {
    setView(v);
    // if global, clear veh; else ensure one is chosen in parent
    if (v === "global") onSelectVehicle(null);
    onRenegotiate?.();
  };

  const userCmd = (cmd) => {
    if (view === "global") return alert("Switch to FRONT/TOP and pick a vehicle.");
    if (!selectedVehicle) return alert("Select a vehicle first.");
    sendCommand({ cmd, veh_id: selectedVehicle });
  };

  return (
    <aside className="w-72 bg-gray-800 border-l border-gray-700 flex flex-col p-3 overflow-y-auto">
      <h2 className="text-lg font-semibold mb-3 text-center">Control Panel</h2>

      {/* View */}
      <div className="mb-3">
        <label className="block text-sm mb-1 text-gray-400">View</label>
        <div className="grid grid-cols-3 gap-2">
          {["front", "top", "global"].map((v) => (
            <button key={v}
              onClick={() => clickView(v)}
              className={`py-1 rounded text-sm ${view === v ? "bg-indigo-600" : "bg-gray-700 hover:bg-gray-600"}`}>
              {v.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Vehicle */}
      <div className="mb-3">
        <label className="block text-sm mb-1 text-gray-400">
          Vehicle {view === "global" && <span className="text-gray-400">(disabled)</span>}
        </label>
        <select
          value={selectedVehicle || ""}
          onChange={pickVeh}
          className="w-full bg-gray-700 text-white rounded px-2 py-1 disabled:opacity-50"
          disabled={view === "global"}>
          <option value="">Select vehicle</option>
          {vehicles.map((v) => (
            <option key={v} value={v}>Vehicle {v}</option>
          ))}
        </select>
      </div>

      {/* Goal */}
      <div className="mb-4">
        <label className="block text-sm mb-1 text-gray-400">Goal (for AI panel later)</label>
        <input
          type="text"
          placeholder="e.g. Overtake and return"
          value={goalInput}
          onChange={(e) => setGoalInput(e.target.value)}
          className="w-full bg-gray-700 text-white rounded px-2 py-1"
        />
        <button
          onClick={() => onGoalChange(goalInput)}
          className="mt-2 w-full text-sm bg-blue-600 hover:bg-blue-700 py-1 rounded">
          Save Goal
        </button>
      </div>

      {/* Manual Controls */}
      {view !== "global" && (
        <div className="mb-3 border-t border-gray-600 pt-2">
          <h3 className="text-sm font-semibold mb-2 text-center text-gray-300">Manual Controls</h3>
          <div className="grid grid-cols-2 gap-2">
            <button onClick={() => userCmd("speed_up")} className="bg-green-600 hover:bg-green-700 py-1 rounded text-sm">Speed ↑</button>
            <button onClick={() => userCmd("slow_down")} className="bg-yellow-600 hover:bg-yellow-700 py-1 rounded text-sm">Speed ↓</button>
            <button onClick={() => userCmd("lane_left")} className="bg-blue-600 hover:bg-blue-700 py-1 rounded text-sm">Lane ←</button>
            <button onClick={() => userCmd("lane_right")} className="bg-blue-600 hover:bg-blue-700 py-1 rounded text-sm">Lane →</button>
            <button onClick={() => userCmd("brake")} className="bg-red-700 hover:bg-red-800 py-1 rounded text-sm">Brake</button>
            <button onClick={() => userCmd("release")} className="bg-gray-600 hover:bg-gray-700 py-1 rounded text-sm">Release</button>
          </div>
        </div>
      )}

      {/* Spacer */}
      <div className="mt-auto" />
    </aside>
  );
}