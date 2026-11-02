import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run build` writes dist/, which `qorfinder serve` serves.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true },
});
