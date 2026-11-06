import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // three.js (3D sector view) is a separate, lazily loaded chunk of ~560 kB.
  build: { outDir: '../Backend/static', emptyOutDir: true, chunkSizeWarningLimit: 700 },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
})
