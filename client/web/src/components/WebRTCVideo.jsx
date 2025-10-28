import React, { forwardRef } from "react";

const WebRTCVideo = forwardRef(function WebRTCVideo(_, ref) {
  return (
    <video
      ref={ref}
      autoPlay
      playsInline
      muted
      className="w-full h-full object-contain bg-black select-none"
    />
  );
});

export default WebRTCVideo;