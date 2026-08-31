import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Proxies API calls to the FastAPI backend during development, so the
// frontend always uses the same relative paths ("/api/v1/...") in dev
// and in production -- no per-environment base-URL branching anywhere
// in the app code.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/healthz': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/readyz': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
