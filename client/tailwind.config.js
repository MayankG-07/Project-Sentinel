/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{js,jsx}',
    './components/**/*.{js,jsx}',
    './app/**/*.{js,jsx}',
    './src/**/*.{js,jsx}',
  ],
  prefix: "",
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        // Global Design Language Colors
        background: "#020617",
        'background-secondary': "#0B1120",
        surface: "#111827",
        'card-background': "rgba(17,24,39,0.72)", // Translucent for glassmorphism
        'primary-accent': "#38BDF8",
        'secondary-accent': "#0EA5E9",
        success: "#10B981",
        warning: "#F59E0B",
        danger: "#EF4444",
        'text-primary': "#F8FAFC",
        'text-secondary': "#94A3B8",
        border: "rgba(255,255,255,0.08)",

        // shadcn/ui default colors (can map to custom colors or keep separate)
        // Keeping them separate for clarity, but you might want to map them
        // to your custom palette for full consistency.
        // For now, these are placeholders or can be removed if not used.
        // You'd typically map these to your custom colors like:
        // primary: {
        //   DEFAULT: "hsl(var(--primary))", // Example using CSS variables
        //   foreground: "hsl(var(--primary-foreground))",
        // },
        // ...
      },
      fontFamily: {
        sans: ['Inter', 'Geist', 'Satoshi', 'sans-serif'], // Prioritize Inter, fallback to Geist, Satoshi
        inter: ['Inter', 'sans-serif'],
        geist: ['Geist', 'sans-serif'],
        satoshi: ['Satoshi', 'sans-serif'],
      },
      boxShadow: {
        'soft-glow': '0 0 15px rgba(56, 189, 248, 0.4), 0 0 30px rgba(14, 165, 233, 0.2)', // Primary/Secondary accent glow
        'lg-glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)', // Deeper shadow for glass cards
        'md-glass': '0 4px 16px 0 rgba(0, 0, 0, 0.25)',
      },
      backdropBlur: {
        'glass': '14px', // Custom blur for glassmorphism
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        // Custom keyframes for subtle glows, particles, etc.
        'pulse-glow': {
          '0%, 100%': { opacity: 0.7, transform: 'scale(1)' },
          '50%': { opacity: 1, transform: 'scale(1.02)' },
        },
        'fade-in-up': {
          '0%': { opacity: 0, transform: 'translateY(10px)' },
          '100%': { opacity: 1, transform: 'translateY(0)' },
        },
        'floating-grid': {
          '0%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
          '100%': { transform: 'translateY(0px)' },
        }
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        'pulse-glow': 'pulse-glow 3s ease-in-out infinite',
        'fade-in-up': 'fade-in-up 0.5s ease-out forwards',
        'floating-grid': 'floating-grid 10s ease-in-out infinite',
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};