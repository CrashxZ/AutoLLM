// client/web/tailwind.config.js
/**
 * Tailwind CSS Configuration
 * --------------------------
 * Enables Tailwind for the CARLA AI-in-the-Loop web client (Vite + React).
 * Includes dark mode, custom colors, and component scanning for all src files.
 */

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: "media", // or 'class' if you want explicit dark toggle
  theme: {
    extend: {
      colors: {
        panel: "#111827", // gray-900
        panel2: "#1f2937", // gray-800
        border: "#374151", // gray-700
        accent: "#60a5fa", // blue-400
        ok: "#16a34a", // green-600
        warn: "#f59e0b", // amber-500
        err: "#ef4444", // red-500
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui"],
        mono: ["JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
};
