import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/chat": "http://localhost:8000",
      "/upload": "http://localhost:8000",
      "/clear-chat": "http://localhost:8000",
      "/new-session": "http://localhost:8000",
      "/history": "http://localhost:8000",
      "/download-chat": "http://localhost:8000",
      "/dataset": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});

