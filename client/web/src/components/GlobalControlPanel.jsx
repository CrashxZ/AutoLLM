// client/web/src/components/GlobalControlPanel.jsx
/**
 * GlobalControlPanel.jsx
 * -------------------------------------------------
 * Displays overview of all vehicles, their onboard intentions,
 * and the global control approval process.
 *
 * Props:
 * - vehicles: [{ veh_id, telemetry, intent, action, mode, approved }]
 * - onApprove: (veh_id, action) => void
 * - onDeny: (veh_id) => void
 * - onSelectVehicle: (veh_id) => void
 */

import React, { useState } from "react";

export default function GlobalControlPanel({
  vehicles = [],
  onApprove,
  onDeny,
  onSelectVehicle,
}) {
  const [autoApprove, setAutoApprove] = useState(true);
  const [expanded, setExpanded] = useState(null);

  const handleApprove = (veh) => {
    if (onApprove) onApprove(veh.veh_id, veh.action);
  };

  const handleDeny = (veh) => {
    if (onDeny) onDeny(veh.veh_id);
  };

  return (
    <div className="bg-gray-900 text-gray-100 p-4 rounded-2xl shadow-lg flex flex-col gap-3">
      <h2 className="text-xl font-semibold text-indigo-400">
        🌐 Global Control Panel
      </h2>

      {/* Auto Approve */}
      <div className="flex items-center justify-between">
        <label className="font-medium text-sm">Auto-Approve Requests:</label>
        <input
          type="checkbox"
          checked={autoApprove}
          onChange={(e) => setAutoApprove(e.target.checked)}
          className="w-4 h-4 accent-indigo-500"
        />
      </div>

      {/* Vehicle list */}
      <div className="max-h-96 overflow-y-auto space-y-2">
        {vehicles.length === 0 ? (
          <p className="text-gray-500 text-sm">No active vehicles.</p>
        ) : (
          vehicles.map((veh) => (
            <div
              key={veh.veh_id}
              className={`bg-gray-800 rounded-lg p-3 border ${
                veh.approved
                  ? "border-green-500"
                  : veh.intent
                  ? "border-yellow-500"
                  : "border-gray-700"
              }`}
            >
              {/* Vehicle header */}
              <div className="flex justify-between items-center mb-1">
                <div className="flex gap-2 items-center">
                  <span className="font-semibold text-indigo-300">
                    Vehicle {veh.veh_id}
                  </span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full ${
                      veh.mode === "LLM"
                        ? "bg-indigo-700 text-white"
                        : "bg-gray-600 text-gray-200"
                    }`}
                  >
                    {veh.mode || "Rule"}
                  </span>
                </div>

                <button
                  className="text-sm text-blue-400 hover:text-blue-300 underline"
                  onClick={() => onSelectVehicle(veh.veh_id)}
                >
                  Inspect
                </button>
              </div>

              {/* Intent summary */}
              <p className="text-gray-300 text-sm mb-1">
                {veh.intent || "No intent generated yet."}
              </p>

              {/* Action summary */}
              {veh.action && (
                <pre className="bg-gray-900 p-2 rounded-md text-xs text-green-400 overflow-x-auto mb-1">
                  {JSON.stringify(veh.action, null, 2)}
                </pre>
              )}

              {/* Telemetry */}
              {expanded === veh.veh_id && veh.telemetry && (
                <div className="bg-gray-900 rounded-md p-2 mt-1 text-xs text-gray-300">
                  <p>
                    <b>Speed:</b> {veh.telemetry.speed_kmh.toFixed(1)} km/h
                  </p>
                  <p>
                    <b>Lane:</b> {veh.telemetry.lane_id} (
                    {veh.telemetry.lane_type})
                  </p>
                  <p>
                    <b>Distance to center:</b>{" "}
                    {veh.telemetry.distance_to_center.toFixed(2)} m
                  </p>
                  <p>
                    <b>Junction:</b>{" "}
                    {veh.telemetry.is_junction ? "Yes" : "No"}
                  </p>
                </div>
              )}

              {/* Expand / Collapse */}
              <button
                className="text-xs text-gray-400 hover:text-gray-200 mt-1"
                onClick={() =>
                  setExpanded(expanded === veh.veh_id ? null : veh.veh_id)
                }
              >
                {expanded === veh.veh_id ? "▲ Hide Telemetry" : "▼ Show Telemetry"}
              </button>

              {/* Approval buttons */}
              <div className="flex justify-end gap-2 mt-2">
                {!veh.approved && veh.intent && (
                  <>
                    <button
                      className="bg-green-600 hover:bg-green-700 text-sm px-3 py-1 rounded-md"
                      onClick={() => handleApprove(veh)}
                    >
                      ✅ Approve
                    </button>
                    <button
                      className="bg-red-600 hover:bg-red-700 text-sm px-3 py-1 rounded-md"
                      onClick={() => handleDeny(veh)}
                    >
                      ❌ Deny
                    </button>
                  </>
                )}
                {veh.approved && (
                  <span className="text-green-400 text-sm font-semibold">
                    ✅ Approved
                  </span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
