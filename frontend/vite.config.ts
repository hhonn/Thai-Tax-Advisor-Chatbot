import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
    plugins: [tailwindcss(), sveltekit()], server: {
        proxy: {
            // Proxying '/api' requests to your backend
            '/api': {
                target: 'http://127.0.0.1:8000', // Your backend URL
                changeOrigin: true,
                // rewrite: (path) => path.replace(/^\/api/, '') // Optional: removes '/api' before sending
            },
            '/docs': {
                target: 'http://127.0.0.1:8000', // Your backend URL
                changeOrigin: true,
                // rewrite: (path) => path.replace(/^\/docs/, '') // Optional: removes '/docs' before sending
            },
            '/openapi.json': {
                target: 'http://127.0.0.1:8000', // Your backend URL
                changeOrigin: true,
                // rewrite: (path) => path.replace(/^\/openapi.json/, '') // Optional: removes '/openapi.json' before sending
            }
        }
    }
});
