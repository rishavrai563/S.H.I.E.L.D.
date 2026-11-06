import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import '@fontsource-variable/archivo/standard.css'
import '@fontsource/jetbrains-mono/400.css'
import './styles.css'
import './deck.css'
import './deck-bg.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
