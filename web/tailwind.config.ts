import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        lab: {
          bg: "#0b0e13",
          panel: "#121721",
          line: "#232b38",
          text: "#d7dee9",
          dim: "#8592a6",
          accent: "#38bdf8",
          ok: "#22c55e",
          warn: "#eab308",
          bad: "#ef4444",
        },
      },
      fontFamily: { mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"] },
    },
  },
  plugins: [],
} satisfies Config;
