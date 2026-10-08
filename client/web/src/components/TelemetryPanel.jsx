// client/web/src/components/TelemetryPanel.jsx
/**
 * TelemetryPanel Component
 * ------------------------
 * Shows live telemetry for the selected vehicle and a compact table for all vehicles.
 * Also maintains a rolling log and a small speed-over-time chart for the selected vehicle.
 *
 * Props:
 *  - telemetry: { [vehId: string]: Telemetry }
 *    Telemetry shape (as sent by server/server.py):
 *      {
 *        ts, veh_id, pose{ x,y,z,yaw }, vel{ x,y,z }, speed_kmh,
 *        lane_id, lane_type, is_junction, distance_to_center, goal_distance,
 *        lane_change{ state, direction }
 *      }
 *
 * Behavior:
 *  - Renders selected vehicle details (if available).
 *  - Renders a mini speed chart for selected vehicle (last ~200 samples).
 *  - Renders a compact table listing all vehicles.
 *  - Maintains a rolling textual log of the last N frames (default 120).
 */

import { useEffect, useMemo, useRef, useState } from "react";

const MAX_LOG = 120;
const MAX_HISTORY = 200;

export default function TelemetryPanel({ telemetry, selectedVehId }) {
  // Normalize selectedVehId to string key
  const keySel = selectedVehId != null ? String(selectedVehId) : null;

  // Keep a rolling log of lines and speed history for the selected vehicle
  const [log, setLog] = useState([]);
  const [speedHist, setSpeedHist] = useState([]); // [{t, v}]
  const lastTsRef = useRef(0);

  // Extract a stable list of vehicles
  const vehList = useMemo(() => {
    if (!telemetry) return [];
    return Object.values(telemetry)
      .map((v) => v)
      .sort((a, b) => (a.veh_id ?? 0) - (b.veh_id ?? 0));
  }, [telemetry]);

  const selected = useMemo(() => {
    if (!telemetry || keySel == null) return null;
    return telemetry[keySel] || telemetry[selectedVehId] || null; // tolerate numeric or string keys
  }, [telemetry, keySel, selectedVehId]);

  // Rolling log builder
  useEffect(() => {
    if (!vehList.length) return;

    // Use the most recent ts among vehicles as "frame ts"
    const newest = vehList.reduce((m, v) => (v.ts > m ? v.ts : m), 0);
    if (!newest || newest === lastTsRef.current) return;

    lastTsRef.current = newest;

    // Build a short line for the selected vehicle if present, else first vehicle
    const v = selected || vehList[0];
    const line = `[${new Date(v.ts * 1000).toLocaleTimeString()}] veh=${v.veh_id} `
      + `spd=${fmt(v.speed_kmh, 1)}km/h lane=${v.lane_id ?? "-"} `
      + `d2c=${fmt(v.distance_to_center, 2)} LC=${fmtLc(v.lane_change)}`;

    setLog((prev) => {
      const next = [...prev, line];
      if (next.length > MAX_LOG) next.shift();
      return next;
    });

    // Append to speed history for selected vehicle
    if (selected) {
      setSpeedHist((prev) => {
        const next = [...prev, { t: v.ts, v: v.speed_kmh || 0 }];
        if (next.length > MAX_HISTORY) next.shift();
        return next;
      });
    }
  }, [vehList, selected]);

  const [activeSection, setActiveSection] = useState("selected");

  return (
    <div className="w-full bg-gray-900/80 border-t border-gray-700 flex flex-col gap-3 p-3">
      <div className="flex flex-wrap gap-2 text-xs">
        {[
          { id: "selected", label: "Selected Vehicle" },
          { id: "speed", label: "Speed History" },
          { id: "fleet", label: "All Vehicles" },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveSection(tab.id)}
            className={`px-3 py-1 rounded font-semibold ${
              activeSection === tab.id ? "bg-indigo-600 text-white" : "bg-gray-800 text-gray-300 hover:bg-gray-700"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeSection === "selected" && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-3">
          <h3 className="text-sm font-semibold text-gray-200 mb-2">Selected Vehicle</h3>
          {selected ? (
            <>
              <KV k="Vehicle ID" v={selected.veh_id} />
              <KV k="Speed (km/h)" v={fmt(selected.speed_kmh, 1)} />
              <KV k="Lane" v={`${selected.lane_id ?? "-"} (${selected.lane_type ?? "-"})`} />
              <KV k="Junction" v={selected.is_junction ? "Yes" : "No"} />
              <KV k="Dist to Center" v={fmt(selected.distance_to_center, 2)} />
              <KV k="Goal Dist" v={fmt(selected.goal_distance, 1)} />
              <KV k="Lane Change" v={fmtLc(selected.lane_change)} />
              <KV k="Yaw" v={fmt(selected?.pose?.yaw, 1)} />
            </>
          ) : (
            <div className="text-sm text-gray-400">No vehicle selected.</div>
          )}
        </div>
      )}

      {activeSection === "speed" && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-3">
          <h3 className="text-sm font-semibold text-gray-200 mb-2">Speed History (Selected)</h3>
          <MiniChart data={speedHist} />
          <div className="mt-2 text-xs text-gray-400">
            Samples: {speedHist.length}/{MAX_HISTORY}
          </div>
        </div>
      )}

      {activeSection === "fleet" && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 overflow-auto">
          <h3 className="text-sm font-semibold text-gray-200 mb-2">All Vehicles</h3>
          <table className="w-full text-xs">
            <thead className="text-gray-400">
              <tr className="text-left border-b border-gray-700">
                <th className="py-1 pr-2">ID</th>
                <th className="py-1 pr-2">Speed</th>
                <th className="py-1 pr-2">Lane</th>
                <th className="py-1 pr-2">LC</th>
                <th className="py-1 pr-2">Jct</th>
                <th className="py-1 pr-2">d2c</th>
              </tr>
            </thead>
            <tbody>
              {vehList.map((v) => (
                <tr
                  key={v.veh_id}
                  className={`border-b border-gray-800 ${selectedVehId === v.veh_id ? "bg-gray-700/40" : ""}`}
                >
                  <td className="py-1 pr-2">{v.veh_id}</td>
                  <td className="py-1 pr-2">{fmt(v.speed_kmh, 1)}</td>
                  <td className="py-1 pr-2">
                    {v.lane_id ?? "-"} <span className="text-gray-500">({v.lane_type ?? "-"})</span>
                  </td>
                  <td className="py-1 pr-2">{fmtLc(v.lane_change)}</td>
                  <td className="py-1 pr-2">{v.is_junction ? "Y" : "N"}</td>
                  <td className="py-1 pr-2">{fmt(v.distance_to_center, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 max-h-44 overflow-auto">
        <h3 className="text-sm font-semibold text-gray-200 mb-2">
          Rolling Log (last {MAX_LOG})
        </h3>
        <div className="font-mono text-xs leading-5 whitespace-pre-wrap">
          {log.length ? log.map((line, idx) => <div key={idx}>{line}</div>) : (
            <div className="text-gray-400">No data yet…</div>
          )}
        </div>
      </div>
    </div>
  );
}

/** Key/Value line */
function KV({ k, v }) {
  return (
    <div className="flex items-center justify-between text-sm py-0.5">
      <span className="text-gray-400">{k}</span>
      <span className="text-gray-100">{String(v ?? "-")}</span>
    </div>
  );
}

/** MiniChart: simple canvas line chart (no external deps). */
function MiniChart({ data }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const ctx = el.getContext("2d");
    const W = el.width;
    const H = el.height;

    // Clear
    ctx.fillStyle = "#111827"; // gray-900
    ctx.fillRect(0, 0, W, H);

    // Axes
    ctx.strokeStyle = "#374151"; // gray-700
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(30, 10);
    ctx.lineTo(30, H - 20);
    ctx.lineTo(W - 10, H - 20);
    ctx.stroke();

    if (!data || data.length < 2) return;

    const xs = data.map((d) => d.t);
    const ys = data.map((d) => d.v);

    const minX = xs[0];
    const maxX = xs[xs.length - 1];
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys, minY + 1);

    const x2px = (x) =>
      30 + ((x - minX) / (maxX - minX)) * (W - 40);
    const y2px = (y) => {
      const t = (y - minY) / (maxY - minY);
      return 10 + (1 - t) * (H - 30);
    };

    // Gridlines
    ctx.strokeStyle = "#1f2937"; // gray-800
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);
    for (let i = 1; i <= 3; i++) {
      const gy = 10 + (i / 4) * (H - 30);
      ctx.beginPath();
      ctx.moveTo(30, gy);
      ctx.lineTo(W - 10, gy);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Line
    ctx.strokeStyle = "#60a5fa"; // blue-400
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x2px(xs[0]), y2px(ys[0]));
    for (let i = 1; i < xs.length; i++) {
      ctx.lineTo(x2px(xs[i]), y2px(ys[i]));
    }
    ctx.stroke();

    // Min/Max labels
    ctx.fillStyle = "#9ca3af"; // gray-400
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillText(`${minY.toFixed(1)}`, 4, y2px(minY));
    ctx.fillText(`${maxY.toFixed(1)}`, 4, y2px(maxY));
  }, [data]);

  return (
    <canvas
      ref={canvasRef}
      width={420}
      height={140}
      className="w-full h-36 bg-gray-900 rounded"
    />
  );
}

function fmt(n, d = 1) {
  if (n == null || Number.isNaN(n)) return "-";
  const f = Number(n);
  return Number.isFinite(f) ? f.toFixed(d) : "-";
}

function fmtLc(lc) {
  if (!lc) return "IDLE";
  const s = lc.state || "IDLE";
  const d = lc.direction ? ` ${String(lc.direction).toUpperCase()}` : "";
  return `${s}${d}`;
}
