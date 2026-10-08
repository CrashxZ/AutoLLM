// client/web/vite.config.js
/**
 * Vite Configuration
 * ------------------
 * Configures Vite for the CARLA AI-in-the-Loop React + Tailwind web client.
 * - React plugin for JSX transformation
 * - Auto port detection
 * - Fast refresh enabled
 * - Public base path auto-handled (Vercel-compatible)
 */

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",   // allow LAN access
    port: 5173,        // default Vite dev port
    strictPort: false, // auto-pick another if busy
    open: true,        // auto open browser
    proxy: {
      "/ws_ui": {
        target: "http://localhost:8000",
        ws: true,
      },
      "/frame": {
        target: "http://localhost:8000",
      },
      "/video": {
        target: "http://localhost:8000",
      },
      "/config": {
        target: "http://localhost:8000",
      },
      "/reset": {
        target: "http://localhost:8000",
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    target: "esnext",
  },
  resolve: {
    alias: {
      "@": "/src",
    },
  },
});
