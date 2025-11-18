// client/web/src/hooks/useWebSocket.js
/**
 * useWebSocket
 * -------------
 * React hook for talking to the FastAPI WebSocket (server/server.py @ /ws).
 *
 * Responsibilities:
 *  - Maintain a resilient WS connection with auto-reconnect + heartbeat.
 *  - Expose real-time telemetry + vehicles list.
 *  - Queue outgoing commands until the socket is open.
 *  - Surface minimal connection status for UI.
 *
 * Usage:
 *   const { telemetry, vehicles, status, sendCommand, connect, disconnect } =
 *     useWebSocket({ url: "ws://localhost:8000/ws", onOpen, onClose });
 *
 * Commands (examples):
 *   sendCommand({ cmd: "select", veh_id: 123 });
 *   sendCommand({ cmd: "speed", veh_id: 123, kmh: 60 });
 *   sendCommand({ cmd: "lane",  veh_id: 123, dir: "left" }); // or "right"
 *   sendCommand({ cmd: "brake", veh_id: 123 });
 *   sendCommand({ cmd: "release", veh_id: 123 });
 *   sendCommand({ cmd: "mode", control_mode: "LLM" }); // or "USER"
 *   sendCommand({ cmd: "intent", intent: { see server docs } });
 */

import { useCallback, useEffect, useRef, useState } from "react";

const DEFAULT_HEARTBEAT_MS = 8_000;
const MAX_BACKOFF_MS = 12_000;

export default function useWebSocket({
  url,
  onOpen,
  onClose,
  heartbeatMs = DEFAULT_HEARTBEAT_MS,
} = {}) {
  const [status, setStatus] = useState("idle"); // idle|connecting|open|closed|error
  const [telemetry, setTelemetry] = useState({});
  const [vehicles, setVehicles] = useState([]);
  const wsRef = useRef(null);
  const hbTimerRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const backoffRef = useRef(500);
  const pendingQueueRef = useRef([]); // messages queued until OPEN

  const clearTimers = () => {
    if (hbTimerRef.current) {
      clearInterval(hbTimerRef.current);
      hbTimerRef.current = null;
    }
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  };

  const closeSocket = useCallback(() => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close(1000, "client closing");
      }
    } catch {}
    wsRef.current = null;
    clearTimers();
    setStatus("closed");
    onClose && onClose();
  }, [onClose]);

  const scheduleReconnect = useCallback(() => {
    clearTimers();
    const delay = Math.min(backoffRef.current, MAX_BACKOFF_MS);
    reconnectTimerRef.current = setTimeout(() => {
      connect();
      backoffRef.current = Math.min(backoffRef.current * 1.7, MAX_BACKOFF_MS);
    }, delay);
  }, []);

  const startHeartbeat = useCallback(() => {
    clearTimers();
    hbTimerRef.current = setInterval(() => {
      try {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        wsRef.current.send(JSON.stringify({ type: "ping", t: Date.now() }));
      } catch {}
    }, heartbeatMs);
  }, [heartbeatMs]);

  const flushQueue = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    while (pendingQueueRef.current.length) {
      const msg = pendingQueueRef.current.shift();
      try {
        wsRef.current.send(JSON.stringify(msg));
      } catch (e) {
        console.warn("[WS] send failed from queue:", e);
        break;
      }
    }
  }, []);

  const handleMessage = useCallback((ev) => {
  let data = null;
  try {
    data = JSON.parse(ev.data);
  } catch {
    return;
  }
  const t = data?.type;

  if (t === "hello") {
    // server sends { veh_ids, control_mode, selected_veh_id }
    const ids = Array.isArray(data.veh_ids) ? data.veh_ids.map(Number) : [];
    if (ids.length) setVehicles(ids);
    // optional: you could stash selected_veh_id here if your UI needs it
    backoffRef.current = 500; // reset backoff on successful handshake
    return;
  }

  // if (t === "veh_list") {
  //   // one-shot list after /config
  //   const ids = Array.isArray(data.veh_ids) ? data.veh_ids.map(Number) : [];
  //   if (ids.length) setVehicles(ids);
  //   return;
  // }

  if (t === "telemetry") {
    // server sends { payload || vehicles, veh_ids }
    const tel = data?.payload || data?.vehicles || {};
    setTelemetry(tel);

    // Prefer veh_ids if present, otherwise derive from payload keys
    let ids = [];
    if (Array.isArray(data?.veh_ids)) {
      ids = data.veh_ids.map(Number);
    } else {
      ids = Object.keys(tel).map((k) => Number(k)).filter((n) => !Number.isNaN(n));
    }

    if (ids.length) {
      setVehicles((prev) => {
        const set = new Set([...(prev || []), ...ids]);
        return Array.from(set).sort((a, b) => a - b);
      });
    }
    return;
  }

  if (t === "veh_list") {
    const v = Array.isArray(data.veh_ids) ? data.veh_ids : [];
    setVehicles(v);
    if (!v.length) setTelemetry({});
    backoffRef.current = 500;
    return;
  }
  if (t === "reset") {
    setVehicles([]);           // <-- clears dropdown
    setTelemetry({});
    return;
  }

  if (t === "selected") {
    window.dispatchEvent(new CustomEvent("ws-selected", { detail: { veh_id: Number(data.veh_id) }}));
    return;
  }

  if (t === "ack" || t === "nack") {
    // optional toast/log
    return;
  }

  if (t === "error") {
    console.warn("[WS] error:", data.error);
    return;
  }

  if (t === "approval") {
    // Global coordinator approval/plan message (if used)
    window.dispatchEvent(new CustomEvent("ws-approval", { detail: data }));
    return;
  }
  if (t === "mec_decision") {
    window.dispatchEvent(new CustomEvent("ws-mec-decision", { detail: data }));
    return;
  }
  }, []);


  const connect = useCallback(() => {
    if (!url) {
      console.error("useWebSocket: missing url");
      return;
    }
    try {
      setStatus("connecting");
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("open");
        onOpen && onOpen();
        startHeartbeat();
        flushQueue();
      };

      ws.onmessage = handleMessage;

      ws.onerror = () => {
        setStatus("error");
      };

      ws.onclose = () => {
        setStatus("closed");
        onClose && onClose();
        scheduleReconnect();
      };
    } catch (e) {
      console.error("[WS] connect error:", e);
      setStatus("error");
      scheduleReconnect();
    }
  }, [url, onOpen, onClose, handleMessage, startHeartbeat, flushQueue, scheduleReconnect]);

  const disconnect = useCallback(() => {
    clearTimers();
    // prevent scheduled reconnects after manual disconnect
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    // reset backoff for next manual connect
    backoffRef.current = 500;
    closeSocket();
  }, [closeSocket]);

  const sendCommand = useCallback((payload) => {
    // Normalize common user-friendly aliases to server's expected format
    const norm = normalizeCommand(payload);
    const json = JSON.stringify(norm);
    console.log("[WS] sendCommand:", norm);
    try {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        pendingQueueRef.current.push(norm);
        console.log("[WS] queued (socket not open)");
        // try a quick connect if not connected
        if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
          connect();
        }
        return;
      }
      wsRef.current.send(json);
    } catch (e) {
      console.warn("[WS] send failed:", e);
    }
  }, [connect]);

  // Auto-connect on mount; cleanup on unmount
  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  return {
    status,
    telemetry,
    vehicles,
    sendCommand,
    connect,
    disconnect,
  };
}

/* ---------------- helpers ---------------- */

function normalizeCommand(cmd) {
  if (!cmd || typeof cmd !== "object") return { cmd: "noop" };

  // Map "lane_left"|"lane_right" -> {cmd:"lane", dir:"left"|"right"}
  if (cmd.cmd === "lane_left") {
    return { cmd: "lane", veh_id: cmd.veh_id, dir: "left" };
  }
  if (cmd.cmd === "lane_right") {
    return { cmd: "lane", veh_id: cmd.veh_id, dir: "right" };
  }

  // Map "speed_up"/"slow_down" to absolute speed changes requires context.
  // Here we let the server-side planner adjust if needed, or client can send explicit "speed" later.
  if (cmd.cmd === "speed_up") {
    return { cmd: "intent", intent: { ego_veh_id: cmd.veh_id, ego_action: "speed up" } };
  }
  if (cmd.cmd === "slow_down") {
    return { cmd: "intent", intent: { ego_veh_id: cmd.veh_id, ego_action: "slow down" } };
  }

  // Pass-through common server-native commands:
  // select, speed {kmh}, lane {dir}, brake, release, mode {control_mode}, intent {…}
  if (cmd.cmd === "mode" && cmd.control_mode) {
    // Ensure server expects USER|LLM
    const m = String(cmd.control_mode || "").toUpperCase();
    return { cmd: "mode", control_mode: m === "USER" ? "USER" : "LLM" };
  }

  return cmd;
}
