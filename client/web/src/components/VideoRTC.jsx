// client/web/src/components/VideoRTC.jsx
import { useEffect, useRef } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function VideoRTC({ view, vehId }) {
  const videoRef = useRef(null);

  useEffect(() => {
    let pc;

    async function start() {
      pc = new RTCPeerConnection();

      pc.ontrack = (event) => {
        if (videoRef.current) {
          videoRef.current.srcObject = event.streams[0];
        }
      };

      // Create a data-only local stream (no audio/video send)
      const offer = await pc.createOffer({
        offerToReceiveAudio: false,
        offerToReceiveVideo: true,
      });
      await pc.setLocalDescription(offer);

      const params = new URLSearchParams();
      params.set("view", view);
      if (vehId && (view === "front" || view === "top")) {
        params.set("veh_id", String(vehId));
      }

      const resp = await fetch(`${API_BASE}/webrtc/offer?${params.toString()}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
        }),
      });
      const answer = await resp.json();
      await pc.setRemoteDescription(answer);
    }

    start();

    return () => {
      if (pc) pc.close();
    };
  }, [view, vehId]);

  return (
    <video
      ref={videoRef}
      autoPlay
      playsInline
      muted
      className="object-contain w-full h-full select-none"
    />
  );
}