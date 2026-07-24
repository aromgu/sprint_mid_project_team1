import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Reuse the repository's MockData directory without duplicating fixtures.
  publicDir: "../MockData",
  server: {
    proxy: {
      "/api": {
        target: process.env.VITE_BACKEND_TARGET || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
