import React, { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const MAX_DECISIONS = 60;
const POSTURE_OPTIONS = [
  { value: "strict", label: "Strict" },
  { value: "balanced", label: "Balanced" },
  { value: "relaxed", label: "Relaxed" },
];
const POSTURE_DESCRIPTIONS = {
  strict: "Zero-risk tolerance; override plans that might conflict.",
  balanced: "Permit maneuvers with manageable risk and mitigation.",
  relaxed: "Favor flow; only reject if an immediate collision is likely.",
};

export default function MecPanel() {
  const [decisions, setDecisions] = useState([]);
  const [safetyPosture, setSafetyPosture] = useState("strict");
  const [savingPosture, setSavingPosture] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [historyResp, cfgResp] = await Promise.all([
          fetch(`${API_BASE}/mec/history`),
          fetch(`${API_BASE}/mec/config`),
        ]);
        if (!cancelled && historyResp.ok) {
          const data = await historyResp.json();
          if (Array.isArray(data?.decisions)) {
            setDecisions(data.decisions.slice(0, MAX_DECISIONS));
          }
        }
        if (!cancelled && cfgResp.ok) {
          const cfg = await cfgResp.json();
          if (cfg?.safety_posture) {
            setSafetyPosture(cfg.safety_posture);
          }
        }
      } catch {}
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const handler = (evt) => {
      const detail = evt.detail;
      if (!detail) return;
      setDecisions((prev) => [detail, ...prev].slice(0, MAX_DECISIONS));
    };
    window.addEventListener("ws-mec-decision", handler);
    return () => window.removeEventListener("ws-mec-decision", handler);
  }, []);

  const changePosture = async (value) => {
    if (savingPosture || value === safetyPosture) return;
    setSavingPosture(true);
    try {
      const resp = await fetch(`${API_BASE}/mec/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ safety_posture: value }),
      });
      if (resp.ok) {
        const data = await resp.json();
        if (data?.safety_posture) {
          setSafetyPosture(data.safety_posture);
        }
      }
    } catch (err) {
      console.error("Failed to update MEC posture:", err);
    } finally {
      setSavingPosture(false);
    }
  };

  return (
    <section className="w-full bg-gray-900 border-t border-gray-800 flex flex-col h-full">
      <div className="px-4 py-2 bg-gray-800 border-b border-gray-700 flex flex-wrap gap-3 items-center justify-between">
        <div className="text-sm font-semibold">MEC Decision Center</div>
        <div className="flex items-center gap-2 text-xs">
          <label className="text-gray-400">Safety Posture</label>
          <select
            value={safetyPosture}
            onChange={(e) => changePosture(e.target.value)}
            className="bg-gray-700 text-white rounded px-2 py-1 text-xs"
          >
            {POSTURE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          {savingPosture && <span className="text-amber-300">Saving…</span>}
        </div>
      </div>
      <div className="px-4 py-2 text-[11px] text-gray-400 border-b border-gray-800">
        {POSTURE_DESCRIPTIONS[safetyPosture] || ""}
      </div>
      <div className="p-3 flex-1 overflow-auto">
        {decisions.length === 0 ? (
          <div className="text-sm text-gray-400">No MEC decisions yet.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-400 border-b border-gray-800 text-left">
                  <th className="py-2 pr-2">Time</th>
                  <th className="py-2 pr-2">Veh</th>
                  <th className="py-2 pr-2">Decision</th>
                  <th className="py-2 pr-2">Reason</th>
                  <th className="py-2 pr-2">Plan / Model</th>
                </tr>
              </thead>
              <tbody>
                {decisions.map((entry, idx) => (
                  <tr key={`${entry.decision_id || idx}`} className="border-b border-gray-800">
                    <td className="py-1 pr-2 text-gray-300">
                      {entry.ts ? new Date(entry.ts * 1000).toLocaleTimeString() : "—"}
                    </td>
                    <td className="py-1 pr-2 text-gray-200">{entry.veh_id ?? "—"}</td>
                    <td className="py-1 pr-2">
                      <span className={`px-2 py-[1px] rounded ${decisionBadge(entry.decision)}`}>
                        {entry.decision || "allow"}
                      </span>
                    </td>
                    <td className="py-1 pr-2 text-gray-300">{entry.reason || "—"}</td>
                    <td className="py-1 pr-2 text-gray-400">
                      {entry?.plan?.summary || "—"}
                      <span className="ml-2 text-gray-500">{entry.model || ""}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}

function decisionBadge(decision) {
  switch ((decision || "").toLowerCase()) {
    case "allow":
    case "approved":
      return "bg-emerald-900 text-emerald-200";
    case "override":
      return "bg-indigo-900 text-indigo-200";
    case "reject":
      return "bg-red-900 text-red-200";
    default:
      return "bg-gray-700 text-gray-200";
  }
}
