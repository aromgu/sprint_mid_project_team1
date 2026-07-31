import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const additionalHosts = (process.env.VITE_ALLOWED_HOSTS || "")
  .split(",")
  .map(host => host.trim())
  .filter(Boolean);

export default defineConfig({
  plugins: [react()],
  // Reuse the repository's MockData directory without duplicating fixtures.
  publicDir: "../MockData",
  server: {
    // A leading dot allows the ngrok domain and its generated subdomains.
    // Add other temporary demo hosts with VITE_ALLOWED_HOSTS=a.example,b.example.
    allowedHosts: [".ngrok-free.dev", ...additionalHosts],
    proxy: {
      "/api": {
        target: process.env.VITE_BACKEND_TARGET || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
