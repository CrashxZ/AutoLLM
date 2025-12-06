// client/web/src/agents/llmAgent.js
/**
 * LLM Agent for client-side intent generation
 * -------------------------------------------
 * Purpose:
 *   Generate intention + low-level control action from the latest observation and goal.
 *   Default is a rule-based policy; can optionally call OpenAI if an API key is provided.
 *
 * Assumptions:
 *   - This runs in the browser (React app).
 *   - If you pass an API key here, it will be exposed to the browser. Prefer a server proxy
 *     for production if you need to keep keys secret.
 *
 * Usage:
 *   import { proposeAction } from "./agents/llmAgent";
 *   const { intention, action } = await proposeAction(obs, goal, { useLLM: true, apiKey });
 *
 * Observation shape (example subset used here):
 *   {
 *     veh_id: number,
 *     speed_kmh: number,
 *     lane_id: number,
 *     distance_to_center: number,
 *     is_junction: boolean,
 *     other_vehicles?: [{ id, speed_kmh, lane_id } ...],
 *     // optional, attach current view for VLM models
 *     frame_base64?: string
 *   }
 */

const OPENAI_API_URL = "https://api.openai.com/v1/chat/completions";

/* ------------------------------ Utilities ------------------------------ */

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

function sanitizeAction(a = {}) {
  const lane = (a.lane_cmd || "none").toLowerCase();
  const lane_cmd = lane === "left" || lane === "right" ? lane : "none";
  const speed_kmh = Number.isFinite(a.speed_kmh) ? clamp(a.speed_kmh, 0, 140) : 0;
  const brake = !!a.brake;
  return { speed_kmh, lane_cmd, brake };
}

function buildObservationText(observation) {
  const ov = observation?.other_vehicles || [];
  const others = ov
    .slice(0, 6)
    .map(
      (v) =>
        `id:${v.id ?? "?"} lane:${v.lane_id ?? "?"} speed:${Number(v.speed_kmh || 0).toFixed(1)}`
    )
    .join(", ");
  return `
Vehicle state:
- Speed: ${(observation.speed_kmh ?? 0).toFixed(1)} km/h
- Lane ID: ${observation.lane_id ?? "?"}
- Distance to lane center: ${(observation.distance_to_center ?? 0).toFixed(2)} m
- Junction ahead: ${!!observation.is_junction}
- Nearby vehicles (${ov.length}): ${others || "none"}
`.trim();
}

/**
 * Try to recover a JSON object from an LLM free-form response.
 */
function extractJson(text) {
  if (!text) return null;
  // Fast path: already JSON
  try {
    const obj = JSON.parse(text);
    return obj && typeof obj === "object" ? obj : null;
  } catch (_) {
    // Try to slice the first {...} block
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    if (start >= 0 && end > start) {
      const sliced = text.slice(start, end + 1);
      try {
        const obj = JSON.parse(sliced);
        return obj && typeof obj === "object" ? obj : null;
      } catch (__){ /* fallthrough */ }
    }
  }
  return null;
}

/* --------------------------- Rule-based agent --------------------------- */

function ruleBasedAgent(observation, goal) {
  const g = (goal || "").toLowerCase();
  const egoSpeed = Number(observation?.speed_kmh || 0);
  const isJunction = !!observation?.is_junction;

  // Parse simple intent from goal
  let lane_cmd = "none";
  if (g.includes("left")) lane_cmd = "left";
  else if (g.includes("right")) lane_cmd = "right";

  // Speed target heuristic
  const BASE = 45; // km/h baseline cruise
  let target = BASE;

  if (egoSpeed < BASE * 0.9) target = clamp(egoSpeed + 6, 0, 80);
  if (egoSpeed > BASE * 1.15) target = clamp(egoSpeed - 8, 0, 100);

  // Safety: brake at junctions or > 90 km/h
  const brake = isJunction || egoSpeed > 90;

  const action = sanitizeAction({
    speed_kmh: brake ? clamp(egoSpeed - 10, 0, 140) : target,
    lane_cmd,
    brake,
  });

  const intentParts = [];
  if (lane_cmd !== "none") intentParts.push(`prepare ${lane_cmd} lane change`);
  intentParts.push(brake ? "apply gentle brake" : `hold ~${action.speed_kmh.toFixed(0)} km/h`);
  const intention = `Rule-based: ${intentParts.join(", ")}`;

  return { intention, action };
}

/* ---------------------------- OpenAI agent ----------------------------- */

async function openAiAgent(observation, goal, apiKey, model = "gpt-4.1-nano") {
  const messages = [
    {
      role: "user",
      content: [
        {
          type: "text",
          text: `
You are an autonomous driving planner. Given an observation and a high-level goal,
produce an intention and a single low-level control action. (as well as a request to the involved cars if needed)

Goal:
${goal}

${buildObservationText(observation)}

Return STRICT JSON:
{"intention":"...", "action":{"speed_kmh": <number>, "lane_cmd":"left|right|none", "brake": true|false}}
          `.trim(),
        },
      ],
    },
  ];

  // Optional: attach current view for VLMs
  if (observation?.frame_base64) {
    messages[0].content.push({
      type: "image_url",
      image_url: { url: `data:image/jpeg;base64,${observation.frame_base64}` },
    });
  }

  try {
    const res = await fetch(OPENAI_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        messages,
        max_tokens: 220,
        temperature: 0.2,
      }),
    });

    const data = await res.json();
    const text = data?.choices?.[0]?.message?.content || "";

    let parsed = extractJson(text);
    if (!parsed) {
      console.warn("[LLM] Non-JSON response, falling back to rule-based:", text);
      return ruleBasedAgent(observation, goal);
    }

    const intention = String(parsed.intention || "").slice(0, 500);
    const action = sanitizeAction(parsed.action);

    return { intention, action };
  } catch (err) {
    console.error("[LLM] Error:", err);
    return ruleBasedAgent(observation, goal);
  }
}

/* ----------------------------- Public API ------------------------------ */

/**
 * Decide the next control action based on observation + goal.
 * @param {object} observation - latest telemetry snapshot for a vehicle
 * @param {string} goal - short natural-language goal
 * @param {object} config - { useLLM?: boolean, apiKey?: string, model?: string }
 * @returns {Promise<{ intention: string, action: { speed_kmh:number, lane_cmd:string, brake:boolean } }>}
 */
export async function proposeAction(observation, goal, config = {}) {
  const { useLLM = false, apiKey = null, model = "gpt-4.1-nano" } = config;

  if (!useLLM || !apiKey) {
    const out = ruleBasedAgent(observation, goal);
    // Console visibility for debugging
    console.debug("[Agent] Rule-based output:", out);
    return out;
  }

  const out = await openAiAgent(observation, goal, apiKey, model);
  console.debug("[Agent] OpenAI output:", out);
  return out;
}

// Named export of the rule-based policy (handy for unit tests / comparisons)
export const rulePolicy = ruleBasedAgent;
