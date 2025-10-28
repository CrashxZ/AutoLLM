// client/web/src/components/ConfigDrawer.jsx
/**
 * ConfigDrawer
 * ------------
 * Slide-over panel for configuring the CARLA simulation.
 *
 * Server endpoints used:
 *   - GET  /spawns                      -> list of spawn points
 *   - GET  /preview_spawn/{index}.jpg   -> small JPEG thumbnail for a spawn
 *   - POST /config { num_cars, spawn_indices[], initial_speeds[] }
 *   - (Optionally call /reset after /config)
 *
 * Props:
 *   - onClose: () => void
 *   - onApply: (cfg) => Promise<void>   (parent usually POSTs /config)
 *
 * Local-only settings (saved in localStorage):
 *   - snapshot_rate_hz (number)
 *   - llm_backend: "LOCAL" | "OPENAI"
 *   - openai_api_key (string)
 */

import { useEffect, useMemo, useState } from "react";

const API_BASE = "http://localhost:8000";

export default function ConfigDrawer({ onClose, onApply }) {
  const [loading, setLoading] = useState(false);
  const [spawns, setSpawns] = useState([]);
  const [numCars, setNumCars] = useState(
    Number(localStorage.getItem("cfg_num_cars") || 2)
  );

  // arrays sized to numCars
  const [spawnIdx, setSpawnIdx] = useState(
    parseCsvArray(localStorage.getItem("cfg_spawn_indices")) || [110, 112]
  );
  const [speeds, setSpeeds] = useState(
    parseCsvArray(localStorage.getItem("cfg_initial_speeds")) || [50, 40]
  );

  // client-only settings
  const [snapHz, setSnapHz] = useState(
    Number(localStorage.getItem("snapshot_rate_hz") || 2)
  );
  const [llmBackend, setLlmBackend] = useState(
    localStorage.getItem("llm_backend") || "LOCAL"
  );
  const [apiKey, setApiKey] = useState(
    localStorage.getItem("openai_api_key") || ""
  );

  // Normalize arrays to numCars
  useEffect(() => {
    setSpawnIdx((prev) => normalizeLen(prev, numCars, (i) => i));
    setSpeeds((prev) => normalizeLen(prev, numCars, () => 50));
  }, [numCars]);

  // Fetch spawns once
  useEffect(() => {
    let ok = true;
    (async () => {
      try {
        setLoading(true);
        const r = await fetch(`${API_BASE}/spawns`);
        const j = await r.json();
        if (!ok) return;
        setSpawns(j.spawns || []);
      } catch (e) {
        console.error(e);
      } finally {
        if (ok) setLoading(false);
      }
    })();
    return () => {
      ok = false;
    };
  }, []);

  const spawnOptions = useMemo(
    () =>
      spawns.map((s) => ({
        value: s.index,
        label: `#${s.index}  (${fmt(s.x, 1)}, ${fmt(s.y, 1)}) yaw=${fmt(
          s.yaw,
          1
        )}`,
      })),
    [spawns]
  );

  const apply = async (andReset = false) => {
    const cfg = {
      num_cars: numCars,
      spawn_indices: spawnIdx.slice(0, numCars),
      initial_speeds: speeds.slice(0, numCars),
    };

    // persist client prefs
    localStorage.setItem("cfg_num_cars", String(numCars));
    localStorage.setItem("cfg_spawn_indices", spawnIdx.join(","));
    localStorage.setItem("cfg_initial_speeds", speeds.join(","));
    localStorage.setItem("snapshot_rate_hz", String(snapHz));
    localStorage.setItem("llm_backend", llmBackend);
    if (apiKey) localStorage.setItem("openai_api_key", apiKey);

    try {
      await onApply?.(cfg); // parent typically POSTs /config
      if (andReset) {
        await fetch(`${API_BASE}/reset`, { method: "POST" });
      }
      onClose?.();
    } catch (e) {
      console.error(e);
      alert("Failed to apply configuration.");
    }
  };

  return (
    <div className="fixed inset-0 z-40 flex">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/60" onClick={onClose} />

      {/* Panel */}
      <div className="relative ml-auto h-full w-full max-w-3xl bg-gray-900 border-l border-gray-700 p-4 overflow-y-auto">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold">Simulation Configuration</h2>
          <button
            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded"
            onClick={onClose}
          >
            ✖ Close
          </button>
        </div>

        {/* General */}
        <section className="mb-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-2">General</h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Number of Cars (1–8)
              </label>
              <input
                type="number"
                min={1}
                max={8}
                value={numCars}
                onChange={(e) =>
                  setNumCars(Math.max(1, Math.min(8, Number(e.target.value))))
                }
                className="w-full bg-gray-800 text-white rounded px-2 py-1"
              />
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Snapshot Rate (Hz)
              </label>
              <input
                type="number"
                min={0}
                max={30}
                step={1}
                value={snapHz}
                onChange={(e) => setSnapHz(Number(e.target.value))}
                className="w-full bg-gray-800 text-white rounded px-2 py-1"
              />
              <p className="text-[11px] text-gray-500 mt-1">
                Client-side frame capture frequency.
              </p>
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">
                LLM Backend
              </label>
              <select
                value={llmBackend}
                onChange={(e) => setLlmBackend(e.target.value)}
                className="w-full bg-gray-800 text-white rounded px-2 py-1"
              >
                <option value="LOCAL">Local (Rule-based)</option>
                <option value="OPENAI">OpenAI API</option>
              </select>
            </div>
          </div>
        </section>

        {/* OpenAI (optional) */}
        {llmBackend === "OPENAI" && (
          <section className="mb-4">
            <h3 className="text-sm font-semibold text-gray-300 mb-2">
              OpenAI Settings
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  API Key
                </label>
                <input
                  type="password"
                  placeholder="sk-..."
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  className="w-full bg-gray-800 text-white rounded px-2 py-1"
                />
                <p className="text-[11px] text-gray-500 mt-1">
                  Stored locally in your browser.
                </p>
              </div>
            </div>
          </section>
        )}

        {/* Vehicles config */}
        <section className="mb-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-2">
            Vehicles
          </h3>

          {loading ? (
            <div className="text-gray-400 text-sm">Loading spawns…</div>
          ) : spawns.length === 0 ? (
            <div className="text-gray-400 text-sm">
              No spawns available (server not ready?).
            </div>
          ) : (
            <div className="space-y-4">
              {Array.from({ length: numCars }).map((_, i) => (
                <div
                  key={i}
                  className="rounded border border-gray-700 p-3 bg-gray-800"
                >
                  <h4 className="text-sm font-semibold text-gray-200 mb-2">
                    Vehicle {i + 1}
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    {/* Spawn select */}
                    <div className="col-span-2">
                      <label className="block text-xs text-gray-400 mb-1">
                        Spawn Point
                      </label>
                      <select
                        value={spawnIdx[i] ?? i}
                        onChange={(e) =>
                          setSpawnIdx((prev) => replaceAt(prev, i, Number(e.target.value)))
                        }
                        className="w-full bg-gray-900 text-white rounded px-2 py-1"
                      >
                        {spawnOptions.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                      {/* Preview */}
                      <div className="mt-2">
                        <img
                          src={`${API_BASE}/preview_spawn/${spawnIdx[i] ?? i}.jpg`}
                          alt={`Spawn ${spawnIdx[i] ?? i}`}
                          className="w-full h-28 object-cover rounded border border-gray-700"
                          onError={(e) => {
                            e.currentTarget.style.display = "none";
                          }}
                        />
                      </div>
                    </div>

                    {/* Speed */}
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">
                        Initial Speed (km/h)
                      </label>
                      <input
                        type="number"
                        min={0}
                        max={150}
                        value={speeds[i] ?? 50}
                        onChange={(e) =>
                          setSpeeds((prev) =>
                            replaceAt(prev, i, Number(e.target.value))
                          )
                        }
                        className="w-full bg-gray-900 text-white rounded px-2 py-1"
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Actions */}
        <div className="mt-6 flex items-center justify-end gap-2">
          <button
            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded"
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded"
            onClick={() => apply(false)}
          >
            Apply
          </button>
        </div>
      </div>
    </div>
  );
}

/* ---------- helpers ---------- */

function normalizeLen(arr, n, filler) {
  const out = Array.isArray(arr) ? [...arr] : [];
  while (out.length < n) out.push(filler(out.length));
  if (out.length > n) out.length = n;
  return out;
}

function replaceAt(arr, idx, value) {
  const out = [...arr];
  out[idx] = value;
  return out;
}

function parseCsvArray(s) {
  if (!s) return null;
  return s.split(",").map((x) => Number(x.trim())).filter((x) => !Number.isNaN(x));
}

function fmt(n, d = 1) {
  if (n == null || Number.isNaN(n)) return "-";
  const f = Number(n);
  return Number.isFinite(f) ? f.toFixed(d) : "-";
}
