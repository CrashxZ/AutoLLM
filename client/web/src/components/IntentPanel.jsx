// client/web/src/components/IntentPanel.jsx
/**
 * IntentPanel.jsx
 * ---------------------------------------------------
 * Displays the current LLM or rule-based agent’s reasoning,
 * shows raw responses, and lets the user approve / override.
 *
 * Props:
 * - observation: latest telemetry (for context)
 * - goal: current goal string
 * - setGoal: (fn) updates the goal text
 * - mode: "USER" | "LLM"
 * - setMode: (fn) toggles control mode
 * - sendAction: (fn) sends approved action to server
 * - apiKey: OpenAI API key (optional)
 */

import React, { useState } from "react";
import { proposeAction } from "../agents/llmAgent";

export default function IntentPanel({
  observation,
  goal,
  setGoal,
  mode,
  setMode,
  sendAction,
  apiKey,
}) {
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [log, setLog] = useState([]);
  const [useLLM, setUseLLM] = useState(mode === "LLM");

  const handleGenerate = async () => {
    if (!goal || !observation) return;
    setLoading(true);
    const start = Date.now();
    try {
      const res = await proposeAction(observation, goal, {
        useLLM,
        apiKey,
      });
      setResponse(res);
      const entry = {
        ts: new Date().toISOString(),
        duration: ((Date.now() - start) / 1000).toFixed(2),
        goal,
        mode: useLLM ? "LLM" : "Rule",
        result: res,
      };
      setLog((prev) => [entry, ...prev.slice(0, 20)]);
    } catch (err) {
      console.error("Error in generate:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSend = () => {
    if (response?.action) {
      sendAction(response.action);
    }
  };

  const handleOverride = () => {
    const custom = prompt(
      "Enter manual override (JSON):",
      JSON.stringify(response?.action || { speed_kmh: 30, lane_cmd: "none", brake: false })
    );
    try {
      const parsed = JSON.parse(custom);
      sendAction(parsed);
    } catch (e) {
      alert("Invalid JSON.");
    }
  };

  return (
    <div className="bg-gray-900 text-gray-100 p-4 rounded-2xl shadow-md flex flex-col gap-3">
      <h2 className="text-xl font-semibold text-indigo-400">AI Intent Panel</h2>

      {/* Mode Toggle */}
      <div className="flex justify-between items-center">
        <label className="font-medium">Control Mode:</label>
        <select
          className="bg-gray-800 px-2 py-1 rounded-md"
          value={useLLM ? "LLM" : "Rule"}
          onChange={(e) => {
            const val = e.target.value === "LLM";
            setUseLLM(val);
            setMode(val ? "LLM" : "USER");
          }}
        >
          <option value="Rule">Rule-Based</option>
          <option value="LLM">LLM Agent</option>
        </select>
      </div>

      {/* Goal input */}
      <div>
        <label className="block text-sm mb-1">Current Goal:</label>
        <textarea
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          rows={2}
          placeholder="e.g., Keep safe distance and stay in middle lane"
          className="w-full bg-gray-800 p-2 rounded-md text-sm resize-none"
        />
      </div>

      <button
        disabled={loading}
        onClick={handleGenerate}
        className="w-full bg-indigo-500 hover:bg-indigo-600 py-2 rounded-md text-sm font-semibold"
      >
        {loading ? "Generating..." : "Generate Intent & Action"}
      </button>

      {/* Response view */}
      {response && (
        <div className="bg-gray-800 rounded-lg p-3 text-sm">
          <p className="text-indigo-300 font-semibold">Intention:</p>
          <p className="mb-2">{response.intention || "—"}</p>
          <p className="text-indigo-300 font-semibold">Action:</p>
          <pre className="bg-gray-900 p-2 rounded-md overflow-x-auto text-xs">
            {JSON.stringify(response.action, null, 2)}
          </pre>
        </div>
      )}

      <div className="flex gap-2">
        <button
          onClick={handleSend}
          className="flex-1 bg-green-600 hover:bg-green-700 py-1 rounded-md text-sm"
        >
          ✅ Send to Global Control
        </button>
        <button
          onClick={handleOverride}
          className="flex-1 bg-yellow-500 hover:bg-yellow-600 py-1 rounded-md text-sm"
        >
          ✏️ Override
        </button>
      </div>

      {/* Log */}
      <div className="mt-3 max-h-48 overflow-y-auto text-xs">
        <h3 className="font-semibold text-gray-300 mb-1">Recent Actions:</h3>
        {log.length === 0 ? (
          <p className="text-gray-500">No entries yet</p>
        ) : (
          <ul className="space-y-1">
            {log.map((entry, idx) => (
              <li
                key={idx}
                className="bg-gray-800 rounded-md p-2 flex flex-col border border-gray-700"
              >
                <div className="flex justify-between">
                  <span>{entry.ts}</span>
                  <span className="text-gray-400">{entry.mode}</span>
                </div>
                <div className="truncate text-gray-300 text-xs">
                  Goal: {entry.goal}
                </div>
                <div className="text-green-400">
                  {entry.result?.action
                    ? `${entry.result.action.speed_kmh} km/h, ${entry.result.action.lane_cmd}, ${
                        entry.result.action.brake ? "Brake" : "Go"
                      }`
                    : "—"}
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
