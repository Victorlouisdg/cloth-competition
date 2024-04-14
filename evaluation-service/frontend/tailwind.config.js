/** @type {import('tailwindcss').Config} */
module.exports = {
  purge: ['./src/**/*.{js,jsx,ts,tsx}', './public/index.html'],
  darkMode: false, // or 'media' or 'class'
  theme: {
    extend: {
      fontFamily: {
        'sans': ['Dm Sans', 'sans-serif'],
        'roboto': ['Roboto', 'sans-serif']
      },
      colors: {
        'dropdown-color': '#333333', // replace with the color you want
      },
    },
  },
  variants: {
    extend: {},
  },
  plugins: [],
}