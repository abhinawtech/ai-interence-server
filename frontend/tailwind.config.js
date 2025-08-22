/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // ChatGPT-style color scheme
        'gpt-gray': {
          50: '#f7f7f8',
          100: '#ececf1',
          200: '#d9d9e3',
          300: '#c5c5d2',
          400: '#acacbe',
          500: '#8e8ea0',
          600: '#757575',
          700: '#40414f',
          800: '#343541',
          900: '#202123',
        },
        'gpt-green': {
          500: '#10a37f',
          600: '#0d8f72',
        },
        'gpt-blue': {
          500: '#6ea7ff',
          600: '#4d90fe',
        }
      },
      fontFamily: {
        'inter': ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-dots': 'pulse 1.5s ease-in-out infinite',
        'fade-in': 'fadeIn 0.2s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        }
      }
    },
  },
  plugins: [],
}