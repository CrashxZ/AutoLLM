// client/web/src/components/IntentGeneratorPanel.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

/**
 * IntentGeneratorPanel
 * --------------------
 * For each enabled vehicle, every `periodSec` seconds:
 *   - Pulls latest top-down JPG: GET /frame/top/{veh_id}.jpg
 *   - Reads telemetry (speed_kmh, lane_id)
 *   - Calls OpenAI with: { goal, speed_kmh, lane_id, veh_id, top_image_b64 }
 *   - Displays the returned intent+request JSON.
 *
 * Props:
 *   - vehicles: number[]
 *   - telemetry: Record<string, { veh_id:number, speed_kmh:number, lane_id:number, ... }>
 *   - goal: string
 *   - periodSec?: number (default 10)
 */
export default function IntentGeneratorPanel({
  vehicles = [],
  telemetry = {},
  goal = "",
  periodSec = 10,
}) {
  const [apiKey, setApiKey] = useState(localStorage.getItem("openai_api_key") || "");
  const [enabled, setEnabled] = useState({}); // { [vehId]: boolean }
  const [rows, setRows] = useState({});       // { [vehId]: { topB64, ctx, intent, ts, error } }
  const [busy, setBusy] = useState(false);
  const timerRef = useRef(null);

  useEffect(() => {
    if (apiKey) localStorage.setItem("openai_api_key", apiKey);
  }, [apiKey]);

  // Clear interval on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  // Rebuild interval when toggles/period change
  useEffect(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (!apiKey) return; // do nothing until key exists

    timerRef.current = setInterval(() => {
      runCycle();
    }, Math.max(2, periodSec) * 1000);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiKey, periodSec, enabled, vehicles, telemetry, goal]);

  const findTele = (vehId) => {
    // server sends string keys
    return telemetry?.[String(vehId)] || null;
  };

  const runCycle = async () => {
    if (busy) return; // gentle guard
    setBusy(true);
    try {
      // sequential to be gentle on API; can parallelize later
      for (const vid of vehicles) {
        if (!enabled[vid]) continue;
        const tele = findTele(vid);
        if (!tele) continue;

        try {
          const topB64 = await fetchTopFrameBase64(vid);
          const ctx = {
            veh_id: vid,
            goal: goal || "Drive safely and keep lane discipline.",
            speed_kmh: Number(tele.speed_kmh ?? 0),
            lane_id: tele.lane_id ?? null,
          };
          const intent = await callOpenAIForIntent(apiKey, ctx, topB64);

          setRows((prev) => ({
            ...prev,
            [vid]: { topB64, ctx, intent, ts: Date.now(), error: null },
          }));
        } catch (err) {
          setRows((prev) => ({
            ...prev,
            [vid]: { ...(prev[vid] || {}), error: String(err) },
          }));
        }
      }
    } finally {
      setBusy(false);
    }
  };

  const toggleVeh = (vid) => {
    setEnabled((e) => ({ ...e, [vid]: !e[vid] }));
  };

  const clearRow = (vid) => {
    setRows((prev) => {
      const n = { ...prev };
      delete n[vid];
      return n;
    });
  };

  return (
    <section className="w-full bg-gray-900 border-t border-gray-800">
      {/* Header / Controls */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <div className="text-sm font-semibold">Vehicular AI (Intent Generator)</div>
        <div className="flex items-center gap-2">
          <input
            type="password"
            placeholder="OpenAI API Key"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            className="bg-gray-700 rounded px-2 py-1 text-xs w-64"
          />
          <span className="text-xs text-gray-400">
            Period: {periodSec}s {busy ? "• running…" : ""}
          </span>
          <button
            onClick={runCycle}
            className="text-xs px-2 py-1 rounded bg-indigo-600 hover:bg-indigo-700"
          >
            Run once
          </button>
        </div>
      </div>

      {/* Per-vehicle rows */}
      <div className="p-3 grid lg:grid-cols-2 gap-3">
        {vehicles.length === 0 && (
          <div className="text-sm text-gray-400">No vehicles available.</div>
        )}
        {vehicles.map((vid) => {
          const row = rows[vid];
          const tele = findTele(vid);
          return (
            <div
              key={vid}
              className="border border-gray-800 rounded-lg overflow-hidden bg-gray-850"
            >
              <div className="flex items-center justify-between px-3 py-2 bg-gray-800">
                <div className="text-sm font-semibold">
                  Vehicle {vid}
                  <span className="ml-2 text-xs text-gray-400">
                    {tele
                      ? `| speed ${tele.speed_kmh?.toFixed?.(1) ?? "-"} km/h | lane ${tele.lane_id ?? "-"}`
                      : "| no telemetry"}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <label className="flex items-center gap-1 text-xs">
                    <input
                      type="checkbox"
                      checked={!!enabled[vid]}
                      onChange={() => toggleVeh(vid)}
                    />
                    Enable
                  </label>
                  <button
                    onClick={() => clearRow(vid)}
                    className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600"
                  >
                    Clear
                  </button>
                </div>
              </div>

              {/* Body */}
              <div className="p-3 grid grid-cols-5 gap-3">
                {/* Top thumb */}
                <div className="col-span-2">
                  <div className="text-xs text-gray-400 mb-1">Top view</div>
                  <div className="aspect-video bg-black/60 rounded flex items-center justify-center overflow-hidden">
                    {row?.topB64 ? (
                      <img
                        src={`data:image/jpeg;base64,${row.topB64}`}
                        alt={`top-${vid}`}
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <span className="text-xs text-gray-500">no frame</span>
                    )}
                  </div>
                </div>

                {/* Intent JSON */}
                <div className="col-span-3">
                  <div className="text-xs text-gray-400 mb-1">
                    Intent (Vehicular Agent)
                  </div>
                  <div className="text-xs bg-black/40 rounded p-2 overflow-auto max-h-40">
                    {row?.intent ? (
                      <pre className="whitespace-pre-wrap">
{JSON.stringify(row.intent, null, 2)}
                      </pre>
                    ) : (
                      <span className="text-gray-500">—</span>
                    )}
                  </div>

                  {/* Meta / Errors */}
                  <div className="mt-2 text-[11px] text-gray-400 flex justify-between">
                    <span>
                      {row?.ts ? `Updated: ${new Date(row.ts).toLocaleTimeString()}` : ""}
                    </span>
                    <span className="text-red-400">{row?.error || ""}</span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

/* ---------------- helpers ---------------- */

async function fetchTopFrameBase64(vehId) {
  const url = `${API_BASE}/frame/top/${vehId}.jpg?${Date.now()}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`top frame ${vehId}: ${resp.status}`);
  const blob = await resp.blob();
  const b64 = await blobToBase64(blob);
  return b64.replace(/^data:image\/jpeg;base64,/, ""); // clean header
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onloadend = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(blob);
  });
}

async function callOpenAIForIntent(apiKey, ctx, topB64) {
  // Strict, compact instruction with JSON-only output
  const sys = [
    "You are the Vehicular Agent for an autonomous car.",
    "Given: goal, current speed_kmh, lane_id and a top-down camera frame (base64).",
    "Return ONLY compact JSON describing a single-step driving intent and a request to nearby vehicles.",
    "Fields: {",
    '  "ego_veh_id": number,',
    '  "ego_action": "lane left"|"lane right"|"speed up"|"slow down"|"keep",',
    '  "reason": string,',
    '  "confidence": number,',
    '  "request": { "to": string[], "ask": string }',
    "}",
  ].join(" ");

  const user = {
    role: "user",
    content: [
      { type: "text", text: `goal: ${ctx.goal}\nspeed_kmh: ${ctx.speed_kmh}\nlane_id: ${ctx.lane_id}\nveh_id: ${ctx.veh_id}` },
      {
        type: "input_image",
        image_url: { url: `data:image/jpeg;base64,${topB64}` },
      },
    ],
  };

  const body = {
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: sys },
      user,
      {
        role: "user",
        content:
          "Respond with ONLY valid minified JSON. No prose. Example: {\"ego_veh_id\":101,\"ego_action\":\"keep\",\"reason\":\"...\",\"confidence\":0.73,\"request\":{\"to\":[],\"ask\":\"none\"}}",
      },
    ],
    temperature: 0.2,
    max_tokens: 300,
  };

  const resp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  const data = await resp.json();
  const txt = (data?.choices?.[0]?.message?.content || "").trim();

  // Best-effort JSON parsing
  try {
    // strip fencing if any
    const cleaned = txt.replace(/^```json|```$/g, "").trim();
    const obj = JSON.parse(cleaned);
    // Ensure required fields + veh id
    return {
      ego_veh_id: Number(obj.ego_veh_id ?? ctx.veh_id),
      ego_action: String(obj.ego_action || "keep"),
      reason: String(obj.reason || ""),
      confidence: Number(obj.confidence ?? 0.5),
      request: {
        to: Array.isArray(obj?.request?.to) ? obj.request.to.map(String) : [],
        ask: String(obj?.request?.ask || "none"),
      },
    };
  } catch (e) {
    return {
      ego_veh_id: ctx.veh_id,
      ego_action: "keep",
      reason: "fallback: parse error",
      confidence: 0.1,
      request: { to: [], ask: "none" },
      _raw: txt,
    };
  }
}