// client/web/src/components/VideoFeed.jsx
import { useEffect, useRef, useState } from "react";

// export default function VideoFeed({ view, selectedVehicle, telemetry, connected }) {
//   const [fps, setFps] = useState(0);
//   const [last, setLast] = useState(Date.now());
//   const imgRef = useRef(null);

//   const base = "http://localhost:8000";
//   const streamUrl =
//     view === "global"
//       ? `${base}/frame/global.jpg?${Date.now()}`
//       : selectedVehicle
//         ? `${base}/frame/${view}/${selectedVehicle}.jpg?${Date.now()}`
//         : "";

//   useEffect(() => {
//     const id = setInterval(() => {
//       const now = Date.now();
//       const dt = now - last;
//       if (dt > 0) setFps(1000 / dt);
//     }, 1000);
//     return () => clearInterval(id);
//   }, [last]);

//   const handleLoad = () => setLast(Date.now());

//   const veh = selectedVehicle && telemetry ? telemetry[selectedVehicle] : null;
//   const laneChange = veh?.lane_change?.state ?? "IDLE";
//   const laneDir = veh?.lane_change?.direction ?? null;
//   const heading = veh?.pose?.yaw ?? 0;

//   const canShow = connected && (view === "global" || selectedVehicle);

//   return (
//     <div className="relative w-full h-full bg-black overflow-hidden flex items-center justify-center">
//       {canShow ? (
//         <>
//           <img
//             ref={imgRef}
//             src={streamUrl}
//             alt="CARLA Video"
//             onLoad={handleLoad}
//             className="object-contain w-full h-full select-none"
//             crossOrigin="anonymous"
//           />
//           <div className="absolute top-2 left-2 text-xs bg-black/60 px-2 py-1 rounded">
//             <div>View: {view.toUpperCase()}</div>
//             {view !== "global" && <div>Vehicle: {selectedVehicle}</div>}
//             <div>FPS: {isFinite(fps) ? fps.toFixed(1) : "—"}</div>
//             {veh?.speed_kmh != null && <div>Speed: {veh.speed_kmh.toFixed(1)} km/h</div>}
//             {veh?.lane_id != null && <div>Lane: {veh.lane_id}</div>}
//             {veh?.is_junction && <div className="text-yellow-400">Junction Ahead</div>}
//           </div>
//           {view !== "global" && (
//             <div className="absolute top-2 right-2 text-xs bg-black/60 px-2 py-1 rounded text-right">
//               <div>
//                 LANE CHANGE:{" "}
//                 <span
//                   className={`${
//                     laneChange === "EXECUTING"
//                       ? "text-yellow-400"
//                       : laneChange === "DONE"
//                       ? "text-green-400"
//                       : "text-gray-400"
//                   }`}
//                 >
//                   {laneChange}
//                 </span>{" "}
//                 {laneDir ? laneDir.toUpperCase() : ""}
//               </div>
//             </div>
//           )}
//           {view !== "global" && (
//             <div className="absolute bottom-2 right-2 bg-black/50 p-2 rounded flex flex-col items-center text-xs">
//               <div className="text-gray-300">Heading</div>
//               <div
//                 className="w-10 h-10 border border-gray-500 rounded-full flex items-center justify-center"
//                 style={{ transform: `rotate(${heading}deg)`, transition: "transform 0.2s ease" }}
//               >
//                 <div className="w-1 h-4 bg-red-500 rounded-full" />
//               </div>
//             </div>
//           )}
//         </>
//       ) : (
//         <div className="text-gray-400 text-sm text-center">
//           {!connected
//             ? "Not connected to server"
//             : view === "global"
//             ? "Global view active"
//             : "Select a vehicle to view video feed"}
//         </div>
//       )}
//     </div>
//   );
// }

// client/web/src/components/VideoFeed.jsx (replace inner rendering)
import VideoRTC from "./VideoRTC";
const API_BASE = "http://localhost:8000";

export default function VideoFeed({ selectedVehicle, telemetry, connected, view }) {
  // WebRTC path:
  const rtcReady =
    connected && (view === "global" || (selectedVehicle && (view === "front" || view === "top")));

  return (
    <div className="relative w-full h-full bg-black overflow-hidden flex items-center justify-center">
      {rtcReady ? (
        <>
          <VideoRTC view={view} vehId={selectedVehicle || undefined} />
          {/* (Keep your HUD overlay here if you want) */}
        </>
      ) : (
        <div className="text-gray-400 text-sm text-center">
          {!connected
            ? "Not connected to server"
            : view === "global"
              ? "Waiting for stream…"
              : "Select a vehicle and view"}
        </div>
      )}
    </div>
  );
}