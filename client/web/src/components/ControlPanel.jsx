import { useEffect, useMemo, useState } from "react";

/**
 * ControlPanel (Sketch-aligned)
 * -----------------------------
 * Layout (top → bottom):
 *  [ TOP | FRONT ]    <-- view toggle
 *  [ Vehicle {id} ]   <-- vehicle chooser
 *  [ MANUAL CONTROL ] <-- 2x3 button grid
 *  [ USER | AI ]      <-- mode toggle (no AI panel yet)
 *
 * Props:
 *  - view: "front" | "top" | "global" (we show only Front/Top in the toggle per sketch)
 *  - onViewChange(v)
 *  - vehicles: number[]
 *  - selectedVehicle, setSelectedVehicle
 *  - mode, onModeChange("USER" | "LLM_LOCAL" | "LLM_API")
 *  - sendCommand({cmd,...})  // we’ll pass veh_id as needed
 */
export default function ControlPanel({
  view = "front",
  onViewChange = () => {},
  vehicles = [],
  selectedVehicle,
  setSelectedVehicle,
  mode = "USER",
  onModeChange = () => {},
  sendCommand = () => {},
}) {
  // keep control_mode persisted (as you already do elsewhere)
  useEffect(() => {
    localStorage.setItem("control_mode", mode);
  }, [mode]);

  // auto-pick first vehicle when list changes
  useEffect(() => {
    if (!vehicles?.length) {
      if (selectedVehicle != null) setSelectedVehicle(null);
      return;
    }
    if (selectedVehicle == null || !vehicles.includes(selectedVehicle)) {
      setSelectedVehicle(vehicles[0]);
    }
  }, [vehicles]); // eslint-disable-line react-hooks/exhaustive-deps

  const disabledNoVeh = useMemo(
    () => !vehicles?.length || selectedVehicle == null,
    [vehicles, selectedVehicle]
  );

  const fire = (cmd) => {
    if (disabledNoVeh) {
      alert("Select a vehicle first.");
      return;
    }
    // Server understands: lane_left/lane_right/speed_up/slow_down/brake/release
    sendCommand({ cmd, veh_id: selectedVehicle });
  };

  const setUser = () => {
    onModeChange("USER");
    // Tell server
    sendCommand({ cmd: "mode", control_mode: "USER" });
  };

  const setAI = () => {
    // keep it simple for now; later you’ll open the AI panel
    onModeChange("LLM_LOCAL"); // or "LLM_API" later
    sendCommand({ cmd: "mode", control_mode: "LLM" });
  };

  return (
    <aside className="w-72 bg-gray-800 border-l border-gray-700 flex flex-col p-3 gap-3">
      {/* TOP | FRONT */}
      <div className="grid grid-cols-2 gap-2">
        {["top", "front"].map((v) => (
          <button
            key={v}
            onClick={() => onViewChange(v)}
            className={`py-2 rounded text-sm font-semibold ${
              view === v ? "bg-indigo-600" : "bg-gray-700 hover:bg-gray-600"
            }`}
          >
            {v.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Vehicle {id} */}
      <div>
        <label className="block text-sm mb-1 text-gray-300">Vehicle</label>
        <select
          value={selectedVehicle ?? ""}
          onChange={(e) => setSelectedVehicle(Number(e.target.value))}
          className="w-full bg-gray-700 text-white rounded px-2 py-2"
          disabled={!vehicles?.length}
        >
          <option value="" disabled={!vehicles?.length}>
            {vehicles?.length ? "Select vehicle" : "No vehicles"}
          </option>
          {vehicles?.map((v) => (
            <option key={v} value={v}>
              Vehicle {v}
            </option>
          ))}
        </select>
      </div>

      {/* MANUAL CONTROL */}
      <div className="border border-gray-700 rounded p-3">
        <div className="text-sm font-semibold text-gray-300 mb-2">MANUAL CONTROL</div>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => fire("speed_up")}
            className="bg-green-600 hover:bg-green-700 py-2 rounded text-sm"
            disabled={disabledNoVeh}
          >
            Speed ↑
          </button>
          <button
            onClick={() => fire("slow_down")}
            className="bg-yellow-600 hover:bg-yellow-700 py-2 rounded text-sm"
            disabled={disabledNoVeh}
          >
            Speed ↓
          </button>
          <button
            onClick={() => fire("lane_left")}
            className="bg-blue-600 hover:bg-blue-700 py-2 rounded text-sm"
            disabled={disabledNoVeh}
          >
            Lane ←
          </button>
          <button
            onClick={() => fire("lane_right")}
            className="bg-blue-600 hover:bg-blue-700 py-2 rounded text-sm"
            disabled={disabledNoVeh}
          >
            Lane →
          </button>
          <button
            onClick={() => fire("brake")}
            className="bg-red-700 hover:bg-red-800 py-2 rounded text-sm"
            disabled={disabledNoVeh}
          >
            Brake
          </button>
          <button
            onClick={() => fire("release")}
            className="bg-gray-600 hover:bg-gray-700 py-2 rounded text-sm"
            disabled={disabledNoVeh}
          >
            Release
          </button>
        </div>
      </div>

      {/* USER | AI */}
      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={setUser}
          className={`py-2 rounded text-sm font-semibold ${
            mode === "USER" ? "bg-indigo-600" : "bg-gray-700 hover:bg-gray-600"
          }`}
        >
          USER
        </button>
        <button
          onClick={setAI}
          className={`py-2 rounded text-sm font-semibold ${
            mode !== "USER" ? "bg-indigo-600" : "bg-gray-700 hover:bg-gray-600"
          }`}
        >
          AI
        </button>
      </div>

      {/* (Future) AI Control Panel placeholder */}
      <div className="mt-2 border border-dashed border-gray-700 rounded p-3 text-xs text-gray-400">
        AI CONTROL PANEL (coming next)
      </div>
    </aside>
  );
}