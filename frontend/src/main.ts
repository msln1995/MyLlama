import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import './styles/global.css'
import { loadConfig, appConfig, applyTheme } from './store'

// Apply cached theme immediately to avoid flash (handles 'auto' -> system dark/light correctly)
applyTheme()

loadConfig().then(() => {
  const app = createApp(App)
  app.use(router)
  app.mount('#app')
})
