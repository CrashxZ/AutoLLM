import { useCallback, useEffect, useRef, useState } from "react";

/**
 * useWebRTC
 * ----------
 * Handles SDP offer/answer to FastAPI /webrtc/offer and wires a remote MediaStream
 * into a provided <video> element. Reconnects on view/veh changes via renegotiate().
 *
 * Server expects:
 *   POST /webrtc/offer?view=front|top|global&veh_id=<id|null>&fps=20&max_bitrate=1200000
 *   body: { sdp, type }
 */
export default function useWebRTC({ apiBase, defaultConstraints = {} } = {}) {
  const videoRef = useRef(null);
  const pcRef = useRef(null);
  const [pcState, setPcState] = useState("new");
  const [error, setError] = useState(null);
  const lastParamsRef = useRef(null);

  const _closePc = useCallback(async () => {
    try {
      if (pcRef.current) {
        pcRef.current.ontrack = null;
        pcRef.current.oniceconnectionstatechange = null;
        pcRef.current.onconnectionstatechange = null;
        await pcRef.current.close();
      }
    } catch {}
    pcRef.current = null;
    setPcState("closed");
  }, []);

  const _createPc = useCallback(() => {
    const pc = new RTCPeerConnection({
      iceServers: [
        { urls: "stun:stun.l.google.com:19302" },
        { urls: "stun:global.stun.twilio.com:3478?transport=udp" },
      ],
    });
    pc.ontrack = (ev) => {
      const [stream] = ev.streams;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(() => {});
      }
    };
    pc.oniceconnectionstatechange = () => {
      setPcState(pc.iceConnectionState || "unknown");
    };
    pc.onconnectionstatechange = () => {
      // keep a slightly higher level state visible
    };
    return pc;
  }, []);

  const _negotiate = useCallback(async ({ view, vehId, fps, maxBitrate }) => {
    setError(null);
    lastParamsRef.current = { view, vehId, fps, maxBitrate };
    try {
      await _closePc();
      const pc = _createPc();
      pcRef.current = pc;

      // Create dummy transceiver to receive only (server will add track)
      pc.addTransceiver("video", { direction: "recvonly" });

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      const params = new URLSearchParams();
      params.set("view", view);
      if (vehId != null) params.set("veh_id", String(vehId));
      if (fps != null) params.set("fps", String(fps));
      if (maxBitrate != null) params.set("max_bitrate", String(maxBitrate));

      const resp = await fetch(`${apiBase}/webrtc/offer?${params.toString()}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
      });
      const data = await resp.json();
      const answer = {
        type: data.type,
        sdp: data.sdp,
      };
      await pc.setRemoteDescription(answer);
      setPcState("connected");
    } catch (e) {
      setError(e);
      setPcState("failed");
    }
  }, [_createPc, _closePc, apiBase]);

  const connect = useCallback(async ({ view = "front", vehId = null, fps, maxBitrate } = {}) => {
    const fpsF = fps ?? defaultConstraints.fps ?? 20;
    const br   = maxBitrate ?? defaultConstraints.maxBitrate ?? 1200000;
    await _negotiate({ view, vehId, fps: fpsF, maxBitrate: br });
  }, [_negotiate, defaultConstraints]);

  const renegotiate = useCallback(async (opts = {}) => {
    const prev = lastParamsRef.current || {};
    await connect({ ...prev, ...opts });
  }, [connect]);

  const disconnect = useCallback(async () => {
    await _closePc();
    try {
      await fetch(`${apiBase}/webrtc/close`, { method: "POST" });
    } catch {}
  }, [_closePc, apiBase]);

  useEffect(() => () => { _closePc(); }, [_closePc]);

  return { videoRef, pcState, error, connect, disconnect, renegotiate };
}