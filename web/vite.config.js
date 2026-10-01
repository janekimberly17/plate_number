import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// base './' makes the built site work at any GitHub Pages address
// (https://<you>.github.io/<repo-name>/) without knowing the repo name.
export default defineConfig({
  plugins: [react()],
  base: './',
})
