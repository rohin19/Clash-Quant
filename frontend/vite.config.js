import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false
      }
    }
  },
  define: {
    // Make API URL configurable for production
    __API_URL__: JSON.stringify(process.env.VITE_API_URL || 'http://207.23.170.7:5000')
  }
})
