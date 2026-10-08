// client/web/src/components/IntentGeneratorPanel.jsx
import React, { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const OPENAI_API_KEY = import.meta.env.VITE_OPENAI_API_KEY || "";
const OPENAI_MODEL = import.meta.env.VITE_OPENAI_MODEL || "gpt-4o-mini";
const SPEED_TOLERANCE_KMH = 2.5;
const STEP_RESEND_INTERVAL_MS = 3000;
const STEP_MAX_RUNTIME_MS = 25000;
const STEP_MAX_RETRIES = 4;

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
  mainVehicleId = null,
  sendIntent = null,
}) {
  const [enabled, setEnabled] = useState({}); // { [vehId]: boolean }
  const [rows, setRows] = useState({}); // { [vehId]: { topB64, ctx, intent, plan, mec, ts, error, goal, latencyMs, latencyHistory } }
  const [goalOverrides, setGoalOverrides] = useState({}); // { [vehId]: string }
  const [busy, setBusy] = useState(false);
  const [copiedVeh, setCopiedVeh] = useState(null);
  const timerRef = useRef(null);
  const busyRef = useRef(false);
  const latestRef = useRef(null);
  const stepTrackersRef = useRef({});
  const DEFAULT_GOAL = "Drive safely and keep lane discipline.";
  const hasApiKey = OPENAI_API_KEY.trim().length > 0;
  const runDisabled = !hasApiKey || busy;
  const sendIntentFn = typeof sendIntent === "function" ? sendIntent : null;
  const mainEnabled = mainVehicleId != null ? !!enabled[mainVehicleId] : false;

  useEffect(() => {
    latestRef.current = {
      vehicles,
      telemetry,
      goal,
      goalOverrides,
      enabled,
      hasApiKey,
      mainVehicleId,
      sendIntentFn,
      rows,
    };
  }, [vehicles, telemetry, goal, goalOverrides, enabled, hasApiKey, mainVehicleId, sendIntentFn, rows]);

  useEffect(() => {
    if (copiedVeh == null) return undefined;
    const t = setTimeout(() => setCopiedVeh(null), 1500);
    return () => clearTimeout(t);
  }, [copiedVeh]);

  useEffect(() => {
    if (mainVehicleId == null) return;
    setEnabled((prev) => {
      if (prev[mainVehicleId]) return prev;
      return { ...prev, [mainVehicleId]: true };
    });
  }, [mainVehicleId]);

  useEffect(() => {
    const allowed = new Set(vehicles.map(String));
    setEnabled((prev) => {
      let changed = false;
      const next = {};
      Object.entries(prev).forEach(([key, value]) => {
        if (allowed.has(key)) {
          next[key] = value;
        } else {
          changed = true;
        }
      });
      if (!changed && Object.keys(next).length === Object.keys(prev).length) return prev;
      return next;
    });
  }, [vehicles]);

  useEffect(() => {
    const allowed = new Set(vehicles.map(String));
    setRows((prev) => {
      let changed = false;
      const next = {};
      Object.entries(prev).forEach(([key, value]) => {
        if (allowed.has(key)) {
          next[key] = value;
        } else {
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  }, [vehicles]);

  useEffect(() => {
    const allowed = new Set(vehicles.map((v) => String(v)));
    Object.keys(stepTrackersRef.current).forEach((key) => {
      if (!allowed.has(key)) {
        delete stepTrackersRef.current[key];
      }
    });
  }, [vehicles]);

  useEffect(() => {
    Object.entries(enabled).forEach(([key, value]) => {
      if (!value && stepTrackersRef.current[key]) {
        delete stepTrackersRef.current[key];
      }
    });
  }, [enabled]);

  useEffect(() => {
    Object.entries(stepTrackersRef.current).forEach(([key, tracker]) => {
      const row = rows[key] ?? rows[Number(key)];
      if (!row?.plan || row.plan.id !== tracker.planId) {
        delete stepTrackersRef.current[key];
      }
    });
  }, [rows]);

  const applyMecDecision = useCallback((vehId, ctx, mecResult) => {
    const normalized = normalizeMecDecision(mecResult);
    const key = String(vehId);
    const decisionLower = (normalized.decision || "").toLowerCase();
    setRows((prev) => {
      const prevRow = prev[key] ?? prev[vehId];
      if (!prevRow) return prev;
      let nextPlan = prevRow.plan;
      if (normalized.plan) {
        const overridePlan = buildPlanFromMecPlan(normalized.plan, ctx);
        if (overridePlan) {
          nextPlan = overridePlan;
        }
      } else if (decisionLower === "reject" && nextPlan) {
        nextPlan = markPlanRejected(nextPlan);
      }
      const next = { ...prev };
      next[key] = {
        ...prevRow,
        plan: nextPlan,
        mec: normalized,
      };
      return next;
    });
    if (normalized.plan || decisionLower === "reject") {
      delete stepTrackersRef.current[key];
    }
  }, []);

  const runCycle = useCallback(async () => {
    const state = latestRef.current;
    if (!state) return;
    if (busyRef.current || !state.hasApiKey) {
      if (!state.hasApiKey) {
        console.warn("IntentGeneratorPanel: set an OpenAI API key before running.");
      }
      return;
    }

    busyRef.current = true;
    setBusy(true);
    try {
      const key = OPENAI_API_KEY;
      const {
        vehicles: vehs,
        enabled: enabledMap,
        telemetry: teleMap,
        goal: panelGoal,
        rows: currentRows,
      } = state;

      for (const vid of vehs) {
        if (!enabledMap?.[vid]) continue;
        const tele = teleMap?.[String(vid)];
        if (!tele) continue;

        const override = (state.goalOverrides?.[vid] || "").trim();
        const goalText = override || panelGoal || DEFAULT_GOAL;
        const existingRow = currentRows?.[vid];
        const planActive = isPlanActive(existingRow?.plan, goalText, existingRow?.mec);
        const startTs = nowMs();

        try {
          const topB64 = await fetchTopFrameBase64(vid);
          const ctx = {
            veh_id: vid,
            goal: goalText,
            speed_kmh: Number(tele.speed_kmh ?? 0),
            lane_id: tele.lane_id ?? null,
          };

          if (planActive) {
            setRows((prev) => {
              const prevRow = prev?.[vid] || {};
              return {
                ...prev,
                [vid]: {
                  ...prevRow,
                  topB64,
                  ctx,
                  ts: Date.now(),
                  error: null,
                  goal: goalText,
                },
              };
            });
            continue;
          }

          const intent = await callOpenAIForIntent(key, ctx, topB64);
          const latencyMs = Math.max(0, nowMs() - startTs);
          const plan = buildPlanFromIntent(intent, ctx, "vehicular_agent");

          setRows((prev) => {
            const prevRow = prev?.[vid] || {};
            const history = Array.isArray(prevRow.latencyHistory)
              ? [...prevRow.latencyHistory.slice(-19), latencyMs]
              : [latencyMs];
            return {
              ...prev,
              [vid]: {
                ...prevRow,
                topB64,
                ctx,
                intent,
                plan,
                mec: {
                  decision: "pending",
                  reason: "Awaiting MEC approval",
                },
                ts: Date.now(),
                error: null,
                goal: goalText,
                latencyMs,
                latencyHistory: history,
              },
            };
          });

          try {
            const mecResult = await requestMecApproval({
              vehId: vid,
              plan,
              ctx,
              intent,
              topFrameB64: topB64,
              telemetry: state.telemetry,
            });
            applyMecDecision(vid, ctx, mecResult);
          } catch (mecErr) {
            console.error("MEC approval failed:", mecErr);
            setRows((prev) => {
              const prevRow = prev?.[vid] || {};
              return {
                ...prev,
                [vid]: {
                  ...prevRow,
                  mec: {
                    decision: "error",
                    reason: String(mecErr),
                  },
                },
              };
            });
          }

          console.log(
            `[IntentGenerator] Vehicle ${vid} plan latency ${latencyMs.toFixed(1)} ms`
          );
        } catch (err) {
          setRows((prev) => {
            const prevRow = prev?.[vid] || {};
            return {
              ...prev,
              [vid]: {
                ...prevRow,
                ts: Date.now(),
                error: String(err),
                goal: goalText,
              },
            };
          });
          console.warn("IntentGeneratorPanel runCycle error:", err);
        }
      }
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [applyMecDecision]);

  // Clear interval on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (!hasApiKey) return;

    const periodMs = Math.max(2, periodSec) * 1000;
    runCycle();
    timerRef.current = setInterval(() => {
      runCycle();
    }, periodMs);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [hasApiKey, periodSec, runCycle]);

  useEffect(() => {
    if (!hasApiKey || !mainEnabled) return;
    runCycle();
  }, [hasApiKey, mainEnabled, runCycle]);

  const updatePlanStep = useCallback((vehId, stepId, mutation) => {
    setRows((prev) => {
      const key = String(vehId);
      const prevRow = prev[key] ?? prev[vehId];
      if (!prevRow?.plan || !Array.isArray(prevRow.plan.steps)) return prev;
      const idx = prevRow.plan.steps.findIndex((s) => s.id === stepId);
      if (idx < 0) return prev;
      const currentStep = prevRow.plan.steps[idx];
      const partial =
        typeof mutation === "function" ? mutation(currentStep) || {} : mutation || {};
      const nextStep = { ...currentStep, ...partial };
      const steps = [...prevRow.plan.steps];
      steps[idx] = nextStep;
      const status = derivePlanStatusFromSteps(steps);
      const nextPlan = {
        ...prevRow.plan,
        steps,
        status,
        completedAt: status === "complete" ? Date.now() : prevRow.plan.completedAt,
      };
      const next = { ...prev };
      next[key] = { ...prevRow, plan: nextPlan };
      return next;
    });
  }, []);

  const issueStepCommand = useCallback(
    (vehId, row, step, tele) => {
      if (!sendIntentFn || !row?.plan) return;
      if (!isMecApproved(row?.mec)) return;
      const payload = buildIntentPayload(vehId, row.plan, step, row.intent, row.mec);
      const message = { cmd: "intent", veh_id: vehId, intent: payload };
      sendIntentFn(message);
      const trackerKey = String(vehId);
      stepTrackersRef.current[trackerKey] = {
        planId: row.plan.id,
        stepId: step.id,
        payload: message,
        issuedAt: Date.now(),
        lastSend: Date.now(),
        retries: 0,
        targetLaneId: step.targetLaneId ?? null,
        targetSpeedKmh: step.targetSpeedKmh ?? null,
        startLaneId: tele?.lane_id ?? null,
        startSpeedKmh: tele?.speed_kmh ?? null,
        action: step.action,
      };
      updatePlanStep(vehId, step.id, {
        status: "running",
        lastIssuedAt: Date.now(),
        retries: 0,
      });
    },
    [sendIntentFn, updatePlanStep]
  );

  useEffect(() => {
    if (!sendIntentFn) return;
    Object.entries(rows).forEach(([key, row]) => {
      const vid = Number(key);
      if (!enabled[vid]) return;
      const plan = row?.plan;
      if (!plan || plan.status === "complete" || plan.status === "failed") {
        return;
      }
      if (!isMecApproved(row?.mec)) {
        return;
      }

      const pending = plan.steps?.find((step) => step.status === "pending");
      const tracker = stepTrackersRef.current[key];
      if (pending) {
        if (tracker && tracker.stepId === pending.id) return;
        const tele = telemetry?.[key] || telemetry?.[String(vid)] || null;
        issueStepCommand(vid, row, pending, tele);
        return;
      }

      const running = plan.steps?.find((step) => step.status === "running");
      if (running && (!tracker || tracker.stepId !== running.id)) {
        const tele = telemetry?.[key] || telemetry?.[String(vid)] || null;
        issueStepCommand(vid, row, running, tele);
      }
    });
  }, [rows, enabled, telemetry, issueStepCommand, sendIntentFn]);

  useEffect(() => {
    if (!sendIntentFn) return;
    const updates = [];
    const now = Date.now();
    Object.entries(stepTrackersRef.current).forEach(([key, tracker]) => {
      const row = rows[key] ?? rows[Number(key)];
      if (!row?.plan) {
        delete stepTrackersRef.current[key];
        return;
      }
      const tele = telemetry?.[key] || telemetry?.[String(key)] || null;
      if (!tele) return;
      const step = row.plan.steps?.find((s) => s.id === tracker.stepId);
      if (!step || step.status !== "running") {
        delete stepTrackersRef.current[key];
        return;
      }

      if (hasStepCompleted(step, tele, tracker)) {
        updates.push({
          vid: Number(key),
          stepId: step.id,
          patch: { status: "done", completedAt: Date.now() },
        });
        delete stepTrackersRef.current[key];
        return;
      }

      if (now - tracker.lastSend >= STEP_RESEND_INTERVAL_MS) {
        if (
          tracker.retries + 1 > STEP_MAX_RETRIES ||
          now - tracker.issuedAt >= STEP_MAX_RUNTIME_MS
        ) {
          updates.push({
            vid: Number(key),
            stepId: step.id,
            patch: {
              status: "failed",
              error: "Timed out waiting for completion.",
              failedAt: Date.now(),
            },
          });
          delete stepTrackersRef.current[key];
        } else {
          tracker.retries += 1;
          tracker.lastSend = now;
          sendIntentFn(tracker.payload);
          updates.push({
            vid: Number(key),
            stepId: step.id,
            patch: { retries: tracker.retries, lastIssuedAt: Date.now() },
          });
        }
      }
    });

    if (updates.length) {
      setRows((prev) => {
        const next = { ...prev };
        updates.forEach(({ vid, stepId, patch }) => {
          const key = String(vid);
          const row = next[key] ?? next[vid];
          if (!row?.plan) return;
          const idx = row.plan.steps.findIndex((s) => s.id === stepId);
          if (idx < 0) return;
          const steps = [...row.plan.steps];
          steps[idx] = { ...steps[idx], ...patch };
          const status = derivePlanStatusFromSteps(steps);
          next[key] = {
            ...row,
            plan: {
              ...row.plan,
              steps,
              status,
              completedAt: status === "complete" ? Date.now() : row.plan.completedAt,
            },
          };
        });
        return next;
      });
    }
  }, [rows, telemetry, sendIntentFn]);

  const findTele = (vehId) => {
    // server sends string keys
    return telemetry?.[String(vehId)] || null;
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
    delete stepTrackersRef.current[String(vid)];
  };

  const updateGoalOverride = (vid, value) => {
    setGoalOverrides((prev) => ({ ...prev, [vid]: value }));
  };

  const clearGoalOverride = (vid) => {
    setGoalOverrides((prev) => {
      const next = { ...prev };
      delete next[vid];
      return next;
    });
  };

  const copyIntent = async (vid) => {
    const row = rows?.[vid];
    if (!row?.intent) return;
    const json = JSON.stringify(row.intent, null, 2);
    try {
      const canClipboard =
        typeof navigator !== "undefined" && navigator.clipboard?.writeText;
      if (canClipboard) {
        await navigator.clipboard.writeText(json);
        setCopiedVeh(vid);
      } else if (typeof window !== "undefined") {
        // fallback prompt ensures user can still grab the payload
        window.prompt("Copy intent JSON:", json);
      }
    } catch (err) {
      console.error("Copy failed:", err);
    }
  };

  return (
    <section className="w-full bg-gray-900 border-t border-gray-800">
      {/* Header / Controls */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <div className="text-sm font-semibold">Vehicular AI (Intent Generator)</div>
        <div className="flex items-center gap-3 flex-wrap justify-end text-xs">
          <span className="text-gray-400">
            Period: {periodSec}s {busy ? "• running…" : ""}
          </span>
          <span className={`px-2 py-[3px] rounded ${hasApiKey ? "bg-gray-700 text-emerald-300" : "bg-gray-700 text-amber-300"}`}>
            {hasApiKey ? "OpenAI key loaded from env" : "Missing VITE_OPENAI_API_KEY"}
          </span>
          <button
            onClick={runCycle}
            disabled={runDisabled}
            className={`text-xs px-2 py-1 rounded bg-indigo-600 ${
              runDisabled ? "opacity-40 cursor-not-allowed" : "hover:bg-indigo-700"
            }`}
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
          const isMain = vid === mainVehicleId;
          const tele = findTele(vid);
          const historyCount = Array.isArray(row?.latencyHistory) ? row.latencyHistory.length : 0;
          const latencyMs = Number.isFinite(row?.latencyMs) ? row.latencyMs : null;
          const avgLatencyMs =
            historyCount > 0
              ? row.latencyHistory.reduce((sum, value) => sum + value, 0) / historyCount
              : null;
          const latencyText =
            latencyMs != null
              ? `Latency: ${(latencyMs / 1000).toFixed(2)}s${
                  historyCount > 1 && avgLatencyMs != null
                    ? ` (avg ${(avgLatencyMs / 1000).toFixed(2)}s)`
                    : ""
                }`
              : null;
          return (
            <div
              key={vid}
              className="border border-gray-800 rounded-lg overflow-hidden bg-gray-850"
            >
              <div className="flex items-center justify-between px-3 py-2 bg-gray-800">
                <div className="text-sm font-semibold">
                  Vehicle {vid}
                  {isMain && <span className="ml-1 text-[11px] text-emerald-400">(main)</span>}
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
                  <div className="text-xs text-gray-400 mb-1">Goal Override</div>
                  <div className="flex items-start gap-2 mb-3">
                    <textarea
                      value={goalOverrides[vid] ?? ""}
                      onChange={(e) => updateGoalOverride(vid, e.target.value)}
                      rows={2}
                      placeholder={`Default: ${goal || DEFAULT_GOAL}`}
                      className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs resize-none"
                    />
                    {(goalOverrides[vid] ?? "") !== "" && (
                      <button
                        onClick={() => clearGoalOverride(vid)}
                        className="text-[11px] px-2 py-1 rounded bg-gray-700 hover:bg-gray-600"
                      >
                        Reset
                      </button>
                    )}
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
                    <div>
                      <div className="text-xs text-gray-400 mb-1">Intent</div>
                      <div className="text-xs bg-black/40 rounded p-2 min-h-[96px]">
                        {row?.intent ? (
                          <>
                            <div className="font-semibold text-indigo-300">
                              {row.intent.ego_action || "—"}
                            </div>
                            <div className="mt-1 text-gray-200">
                              {row.intent.reason || "No reason provided."}
                            </div>
                            <div className="mt-1 text-gray-400">
                              Confidence:{" "}
                              {Number.isFinite(row.intent.confidence)
                                ? row.intent.confidence.toFixed(2)
                                : "n/a"}
                            </div>
                          </>
                        ) : (
                          <span className="text-gray-500">—</span>
                        )}
                      </div>
                    </div>

                    <div>
                      <div className="text-xs text-gray-400 mb-1">Request</div>
                      <div className="text-xs bg-black/40 rounded p-2 min-h-[96px]">
                        {row?.intent ? (
                          <>
                            <div className="text-gray-200">
                              Ask: {row.intent?.request?.ask || "none"}
                            </div>
                            <div className="mt-1 text-gray-400">
                              To:{" "}
                              {Array.isArray(row.intent?.request?.to) &&
                              row.intent.request.to.length
                                ? row.intent.request.to.join(", ")
                                : "none"}
                            </div>
                          </>
                        ) : (
                          <span className="text-gray-500">—</span>
                        )}
                      </div>
                    </div>
                  </div>

                  {row?.plan && (
                    <div className="mt-3">
                      <div className="text-xs text-gray-400 mb-1">Plan</div>
                      <div className="text-xs bg-black/40 rounded p-2 space-y-2">
                        <div className="flex justify-between items-center text-[11px] text-gray-400">
                          <span className="text-gray-200">{row.plan.summary || "LLM plan"}</span>
                          <span className={`px-2 py-[1px] rounded ${planStatusClass(row.plan.status)}`}>
                            {row.plan.status || "pending"}
                          </span>
                        </div>
                        {row?.mec && (
                          <div
                            className={`text-[11px] px-2 py-[3px] rounded ${mecStatusClass(
                              row.mec.decision
                            )}`}
                          >
                            MEC: {formatMecDecision(row.mec)}
                          </div>
                        )}
                        {row.plan.steps?.length ? (
                          <ol className="list-decimal ml-4 space-y-1">
                            {row.plan.steps.map((step) => {
                              const targetText = formatStepTargets(step);
                              return (
                                <li key={step.id} className="text-gray-200">
                                  <div className="flex items-center justify-between gap-2">
                                    <span className="flex-1">
                                      {step.label || describeStep(step)}
                                      {targetText && (
                                        <span className="ml-2 text-gray-400">{targetText}</span>
                                      )}
                                    </span>
                                    <span
                                      className={`text-[11px] font-semibold ${stepStatusClass(step.status)}`}
                                    >
                                      {step.status || "pending"}
                                    </span>
                                  </div>
                                  {step.error && (
                                    <div className="text-[11px] text-red-400">{step.error}</div>
                                  )}
                                </li>
                              );
                            })}
                          </ol>
                        ) : (
                          <div className="text-gray-500">No steps provided.</div>
                        )}
                      </div>
                    </div>
                  )}

                  {row?.intent?._raw && (
                    <div className="mt-2 text-[11px] text-amber-400 bg-amber-900/20 border border-amber-700 rounded p-2">
                      Raw response preserved for debugging.
                    </div>
                  )}

                  {/* Meta / Errors */}
                  <div className="mt-2 text-[11px] text-gray-400 flex flex-wrap gap-2 items-center justify-between">
                    <div className="flex flex-wrap gap-2 items-center">
                      {row?.goal && (
                        <span className="text-gray-300">
                          Goal: <span className="text-gray-100">{row.goal}</span>
                        </span>
                      )}
                      {latencyText && <span>{latencyText}</span>}
                      <span>
                        {row?.ts ? `Updated: ${new Date(row.ts).toLocaleTimeString()}` : ""}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {row?.intent && (
                        <button
                          onClick={() => copyIntent(vid)}
                          className="px-2 py-[3px] rounded bg-gray-700 hover:bg-gray-600 text-gray-200"
                        >
                          {copiedVeh === vid ? "Copied!" : "Copy JSON"}
                        </button>
                      )}
                      <span className="text-red-400">{row?.error || ""}</span>
                    </div>
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

function nowMs() {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return performance.now();
  }
  return Date.now();
}

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
  const sys = [
    "Vehicular agent: output only compact JSON for one car.",
    "Use at most 3 steps to satisfy the goal.",
    'Allowed actions: "lane_left","lane_right","set_speed","speed_up","speed_down","hold","brake".',
    'If another vehicle blocks or is too close, include a request (to ["veh_id"| "unknown"]) such as "slow down" or "yield".',
    'JSON shape: {"ego_veh_id":n,"plan_summary":"...","plan_steps":[{"id":"s1","description":"...","action":"lane_left","target_lane_id":-1,"target_speed_kmh":40}], "ego_action":"lane left","reason":"...","confidence":0.6,"request":{"to":["150"],"ask":"slow down"}}',
  ].join(" ");

  const useImage = Boolean(topB64);
  const userContent = [
    {
      type: "text",
      text: `goal: ${ctx.goal}\nspeed_kmh: ${ctx.speed_kmh}\nlane_id: ${ctx.lane_id}\nveh_id: ${ctx.veh_id}`,
    },
  ];
  if (useImage) {
    userContent.push({
      type: "image_url",
      image_url: { url: `data:image/jpeg;base64,${topB64}` },
    });
  }

  const body = {
    model: OPENAI_MODEL,
    messages: [
      { role: "system", content: sys },
      { role: "user", content: userContent },
      {
        role: "user",
        content:
          'Respond with ONLY minified JSON. Example: {"ego_veh_id":101,"plan_summary":"Shift left and match flow","plan_steps":[{"id":"s1","description":"Move left to faster lane","action":"lane_left","target_lane_id":-2},{"id":"s2","description":"Hold 45 km/h","action":"set_speed","target_speed_kmh":45}],"ego_action":"lane left","reason":"avoid slow traffic","confidence":0.78,"request":{"to":["150"],"ask":"slow down"}}',
      },
    ],
    response_format: { type: "json_object" },
    max_completion_tokens: 220,
  };

  const resp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const errorText = await resp.text();
    throw new Error(`OpenAI ${resp.status}: ${errorText}`);
  }

  const data = await resp.json();
  if (data?.error) {
    throw new Error(`OpenAI error: ${data.error?.message || data.error?.type || "unknown"}`);
  }

  const txt = (data?.choices?.[0]?.message?.content || "").trim();
  if (!txt) {
    throw new Error("OpenAI returned empty content");
  }

  // Best-effort JSON parsing
  try {
    // strip fencing if any
    const cleaned = txt.replace(/```json|```/g, "").trim();
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
      plan_summary: obj.plan_summary || "",
      plan_steps: Array.isArray(obj.plan_steps) ? obj.plan_steps : [],
    };
  } catch (e) {
    return {
      ego_veh_id: ctx.veh_id,
      ego_action: "keep",
      reason: `fallback: ${e.message || "parse error"}`,
      confidence: 0.1,
      request: { to: [], ask: "none" },
      plan_summary: "",
      plan_steps: [],
      _raw: txt,
    };
  }
}

async function requestMecApproval({ vehId, plan, ctx, intent, topFrameB64, telemetry }) {
  const payload = {
    veh_id: vehId,
    plan,
    intent,
    context: {
      goal: ctx.goal,
      speed_kmh: ctx.speed_kmh,
      lane_id: ctx.lane_id,
      telemetry: buildTelemetrySummary(telemetry),
    },
    top_frame_b64: topFrameB64,
  };
  const resp = await fetch(`${API_BASE}/mec/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (resp.status === 404) {
    return {
      decision: "allow",
      reason: "MEC endpoint unavailable — bypassing safety layer.",
      decision_id: null,
    };
  }
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`MEC ${resp.status}: ${txt}`);
  }
  const data = await resp.json();
  return data;
}

function buildTelemetrySummary(telemetry, limit = 5) {
  if (!telemetry) return [];
  const values = Object.values(telemetry)
    .filter(Boolean)
    .sort((a, b) => (a.veh_id ?? 0) - (b.veh_id ?? 0));
  return values.slice(0, limit).map((item) => ({
    veh_id: item.veh_id,
    speed_kmh: item.speed_kmh,
    lane_id: item.lane_id,
    distance_to_center: item.distance_to_center,
    lane_change: item.lane_change,
  }));
}

function buildPlanFromIntent(intent, ctx, source = "vehicular_agent") {
  if (!intent) return null;
  const steps = normalizePlanSteps(intent.plan_steps);
  if (!steps.length) steps.push(fallbackStepFromIntent(intent));
  return createPlanFromSteps(
    ctx,
    steps,
    intent.plan_summary || intent.reason || intent.ego_action || "LLM plan",
    source
  );
}

function createPlanFromSteps(ctx, steps, summary, source = "vehicular_agent", planId) {
  const id = planId || `plan-${ctx.veh_id}-${Date.now()}`;
  return {
    id,
    summary,
    goal: ctx.goal,
    createdAt: Date.now(),
    status: derivePlanStatusFromSteps(steps),
    steps,
    source,
  };
}

function normalizePlanSteps(rawSteps) {
  if (!Array.isArray(rawSteps)) return [];
  return rawSteps
    .map((step, idx) => {
      if (!step) return null;
      const action = sanitizeAction(
        step.action || step.command || step.ego_action || step.type || ""
      );
      const id = String(step.id || step.step_id || `step-${idx + 1}`);
      return {
        id,
        label: step.description || step.summary || step.reason || describeActionFromCode(action),
        action,
        targetLaneId: numberOrNull(step.target_lane_id),
        targetSpeedKmh: numberOrNull(step.target_speed_kmh),
        status: "pending",
      };
    })
    .filter(Boolean);
}

function fallbackStepFromIntent(intent) {
  const action = sanitizeAction(intent?.ego_action || "");
  return {
    id: "step-1",
    label: intent?.reason || intent?.ego_action || "Maintain lane",
    action,
    targetLaneId: numberOrNull(intent?.target_lane_id),
    targetSpeedKmh: numberOrNull(intent?.target_speed_kmh),
    status: "pending",
  };
}

function buildPlanFromMecPlan(mecPlan, ctx) {
  if (!mecPlan) return null;
  const rawSteps = Array.isArray(mecPlan.steps) ? mecPlan.steps : Array.isArray(mecPlan) ? mecPlan : [];
  const steps = normalizePlanSteps(rawSteps);
  if (!steps.length) return null;
  return createPlanFromSteps(ctx, steps, mecPlan.summary || "MEC override plan", "mec");
}

function sanitizeAction(raw) {
  const text = String(raw || "")
    .trim()
    .toLowerCase();
  if (!text) return "hold";
  if (text.includes("lane") && text.includes("left")) return "lane_left";
  if (text.includes("lane") && text.includes("right")) return "lane_right";
  if (text.includes("set") && text.includes("speed")) return "set_speed";
  if (text.includes("speed") && text.includes("up")) return "speed_up";
  if (text.includes("speed") && (text.includes("down") || text.includes("slow"))) return "speed_down";
  if (text.includes("brake") || text.includes("stop")) return "brake";
  return "hold";
}

function derivePlanStatusFromSteps(steps) {
  if (!Array.isArray(steps) || !steps.length) return "complete";
  if (steps.some((s) => s.status === "failed")) return "failed";
  if (steps.every((s) => s.status === "done")) return "complete";
  if (steps.some((s) => s.status === "running")) return "running";
  return "ready";
}

function isPlanActive(plan, goalText, mec) {
  if (!plan || plan.goal !== goalText) return false;
  if (!Array.isArray(plan.steps) || !plan.steps.length) return false;
  if (plan.status === "failed" || plan.status === "complete") return false;
  if (mec && ["reject", "error"].includes((mec.decision || "").toLowerCase())) return false;
  return plan.steps.some((s) => s.status !== "done" && s.status !== "failed");
}

function buildIntentPayload(vehId, plan, step, baseIntent, mec) {
  const stepIndex = plan.steps.findIndex((s) => s.id === step.id);
  return {
    ego_veh_id: vehId,
    ego_action: actionToEgoString(step.action),
    reason: step.label || baseIntent?.reason || plan.summary || "Planner step",
    confidence: baseIntent?.confidence ?? 0.6,
    request: baseIntent?.request || { to: [], ask: "none" },
    target_lane_id: step.targetLaneId ?? null,
    target_speed_kmh: step.targetSpeedKmh ?? null,
    plan_id: plan.id,
    plan_step_id: step.id,
    plan_step_index: stepIndex,
    plan_step_total: plan.steps.length,
    plan_summary: plan.summary,
    mec_decision_id: mec?.decision_id || null,
    mec_decision: mec?.decision || null,
  };
}

function actionToEgoString(action) {
  switch (action) {
    case "lane_left":
      return "lane left";
    case "lane_right":
      return "lane right";
    case "speed_up":
    case "set_speed":
      return "speed up";
    case "speed_down":
      return "slow down";
    case "brake":
      return "brake";
    default:
      return "keep";
  }
}

function describeStep(step) {
  return step?.label || describeActionFromCode(step?.action);
}

function describeActionFromCode(action) {
  switch (action) {
    case "lane_left":
      return "Change to the left lane";
    case "lane_right":
      return "Change to the right lane";
    case "set_speed":
    case "speed_up":
      return "Increase speed";
    case "speed_down":
      return "Reduce speed";
    case "brake":
      return "Brake and hold";
    default:
      return "Maintain lane";
  }
}

function formatStepTargets(step) {
  const parts = [];
  if (Number.isFinite(step?.targetLaneId)) {
    parts.push(`lane ${step.targetLaneId}`);
  }
  if (Number.isFinite(step?.targetSpeedKmh)) {
    parts.push(`${step.targetSpeedKmh} km/h`);
  }
  return parts.length ? `(${parts.join(", ")})` : "";
}

function planStatusClass(status) {
  switch (status) {
    case "complete":
      return "bg-emerald-900 text-emerald-200";
    case "failed":
      return "bg-red-900 text-red-200";
    case "running":
      return "bg-blue-900 text-blue-200";
    default:
      return "bg-gray-700 text-gray-200";
  }
}

function stepStatusClass(status) {
  switch (status) {
    case "done":
      return "text-emerald-300";
    case "running":
      return "text-blue-300";
    case "failed":
      return "text-red-400";
    default:
      return "text-gray-300";
  }
}

function mecStatusClass(decision) {
  switch ((decision || "").toLowerCase()) {
    case "allow":
    case "approved":
      return "bg-emerald-900/60 text-emerald-200";
    case "override":
      return "bg-indigo-900/60 text-indigo-200";
    case "reject":
    case "error":
      return "bg-red-900/50 text-red-200";
    case "pending":
      return "bg-amber-900/40 text-amber-200";
    default:
      return "bg-gray-800 text-gray-200";
  }
}

function formatMecDecision(mec) {
  if (!mec) return "pending";
  const decision = mec.decision || "pending";
  const reason = mec.reason ? ` – ${mec.reason}` : "";
  return `${decision}${reason}`;
}

function normalizeMecDecision(result) {
  if (!result) {
    return { decision: "error", reason: "No MEC response" };
  }
  return {
    decision: (result.decision || "allow").toLowerCase(),
    reason: result.reason || "",
    plan: result.plan || null,
    model: result.model || "",
    ts: result.ts || Date.now() / 1000,
    decision_id: result.decision_id || null,
  };
}

function markPlanRejected(plan) {
  if (!plan) return plan;
  const steps = Array.isArray(plan.steps)
    ? plan.steps.map((step) => ({ ...step, status: "failed" }))
    : [];
  return {
    ...plan,
    steps,
    status: "failed",
  };
}

function isMecApproved(mec) {
  const decision = (mec?.decision || "").toLowerCase();
  return decision === "allow" || decision === "approved" || decision === "override";
}

function hasStepCompleted(step, tele, tracker) {
  if (!step || !tele) return false;
  const action = step.action;
  if (action === "lane_left" || action === "lane_right") {
    const laneChange = tele.lane_change || {};
    if (laneChange.state === "DONE") return true;
    if (
      Number.isFinite(step.targetLaneId) &&
      Number.isFinite(tele.lane_id) &&
      Number(step.targetLaneId) === Number(tele.lane_id) &&
      laneChange.state !== "EXECUTING"
    ) {
      return true;
    }
    if (
      tracker?.startLaneId != null &&
      tele.lane_id != null &&
      Number(tracker.startLaneId) !== Number(tele.lane_id) &&
      laneChange.state !== "EXECUTING"
    ) {
      return true;
    }
    return false;
  }

  if (action === "brake") {
    return (tele.speed_kmh || 0) <= Math.max((tracker?.startSpeedKmh || 0) * 0.25, 5);
  }

  if (action === "set_speed" || action === "speed_up" || action === "speed_down") {
    const target =
      Number.isFinite(step.targetSpeedKmh) && step.targetSpeedKmh != null
        ? step.targetSpeedKmh
        : tracker?.targetSpeedKmh;
    const currentSpeed = Number(tele.speed_kmh || 0);
    if (Number.isFinite(target)) {
      return Math.abs(currentSpeed - target) <= SPEED_TOLERANCE_KMH;
    }
    if (tracker?.startSpeedKmh != null) {
      if (action === "speed_down") {
        return currentSpeed <= tracker.startSpeedKmh - SPEED_TOLERANCE_KMH;
      }
      return currentSpeed >= tracker.startSpeedKmh + SPEED_TOLERANCE_KMH;
    }
    return false;
  }

  return true;
}

function numberOrNull(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}
