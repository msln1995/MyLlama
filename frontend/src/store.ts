import { reactive } from 'vue'
import { getConfig, setTheme as setThemeBackend, setSidebarCollapsed as setSidebarCollapsedBackend, setOnboardingDismissed as setOnboardingDismissedBackend, setDownloadSource as setDownloadSourceBackend, setLanguage as setLanguageBackend, setTrayEnabled as setTrayEnabledBackend, getServerConfig, saveServerConfig as saveServerConfigBackend, saveRemoteChat as saveRemoteChatBackend, type RemoteChatConfig } from './wails'
import type { RemoteChatProfile } from './lib/chatRemote'
import { setLocale, getSystemLanguageClient } from './lib/i18n'

// Theme localStorage key: after the llama-gui → llama-desktop rename the legacy key is read-only fallback
// (preserving theme preference of old installs); writes always use the new key and never delete the old one.
const THEME_KEY = 'llama-desktop-theme'
const LEGACY_THEME_KEY = 'llama-gui-theme'

// Sidebar collapsed-state localStorage key: '0' explicitly expanded, '1'/missing collapsed.
// Same dual-track pattern as theme/tray preferences: UI reads localStorage for a correct
// first frame, while the backend config is only the persistence source (loadConfig
// overwrites it and syncs back to localStorage).
const SIDEBAR_KEY = 'llama-desktop-sidebar-collapsed'

/** Read the persisted theme: prefer the new key, fall back to the pre-rename legacy key, else 'light'. */
export function readStoredTheme(): string {
  return localStorage.getItem(THEME_KEY) || localStorage.getItem(LEGACY_THEME_KEY) || 'light'
}

/**
 * 客户端环境系统主题探测：标准 matchMedia 查询。
 * 在 Android WebView / 桌面端均原生生效，无需后端进程无权执行 shell 命令。
 */
export function getSystemThemeClient(): 'light' | 'dark' {
  if (typeof window !== 'undefined' && window.matchMedia) {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }
  return 'light'
}

/**
 * 将配置的主题值 (auto | light | dark) 解析为真正应用的 HTML data-theme (light | dark)。
 */
export function resolveEffectiveTheme(theme: string): 'light' | 'dark' {
  if (theme === 'auto') {
    return getSystemThemeClient()
  }
  return theme === 'dark' ? 'dark' : 'light'
}

/**
 * 应用主题至 documentElement 的 data-theme 属性。
 * 确保 CSS 选择器 `html[data-theme='dark']` 永不出现接收到 "auto" 导致失效的 BUG。
 */
export function applyTheme(prefTheme?: string): 'light' | 'dark' {
  const theme = prefTheme || appConfig.theme || 'auto'
  const effective = resolveEffectiveTheme(theme)
  document.documentElement.setAttribute('data-theme', effective)
  return effective
}

/** Read the persisted sidebar collapsed state: '0' means explicitly expanded, anything else ('1'/missing) collapsed.
 * First run (key missing) defaults to collapsed to avoid the layout jump of rendering expanded then shrinking. */
export function readStoredSidebarCollapsed(): boolean {
  return localStorage.getItem(SIDEBAR_KEY) !== '0'
}

export const appConfig = reactive({
  theme: readStoredTheme(),
  llamaCppDir: '',
  modelsDir: '',
  // User-chosen download paths: empty means unset (backend falls back to the
  // default llama-cpp/ and LLM-Models directories).
  llamaCppDownloadDir: '',
  modelDownloadDir: '',
  downloadSource: 'hf',
  serverAccessMode: 'local',
  // Optional llama-server API key (bearer token for inference requests); empty
  // means no authentication (backend default). Mirrored from the server config
  // so the Settings input can roll back on save failure.
  serverApiKey: '',
  language: 'auto',
  resolvedLanguage: 'zh' as 'zh' | 'en',
  // Windows system tray toggle: default true (matches the backend loadConfig fallback);
  // only overridden once loadConfig fetches the persisted backend value.
  trayEnabled: true,
  // API-route (headless) mode toggle: default false (matches the backend loadConfig
  // fallback); in GUI mode always false — the headless "Show Main Window" path resets
  // it in the backend before relaunching the GUI, and enabling quits this process.
  apiRouteMode: false,
  // Sidebar collapsed state: read from localStorage for the first frame ('0' expanded,
  // otherwise collapsed) so the sidebar does not render at the wrong width before the
  // async loadConfig returns; defaults to collapsed when the key is missing (matching
  // the backend's preset true fallback).
  sidebarCollapsed: readStoredSidebarCollapsed(),
  // Quick-start checklist dismissed state: default false (visible) until the
  // backend config reports an explicit true (user closed it or completed all
  // steps); same missing-field fallback pattern as apiRouteMode.
  onboardingDismissed: false,
  // LAN remote-chat pairing (phone → PC llama-server): defaults match the
  // backend loadConfig fallback for a config without the remoteChat key
  // (disabled pairing against the llama-server default port 8080); only
  // overridden once loadConfig fetches the persisted backend value.
  remoteChat: { enabled: false, host: '', port: 8080, apiKey: '' } as RemoteChatProfile,
  loaded: false,
})

export async function loadConfig() {
  try {
    const config = await getConfig()
    appConfig.theme = config.theme || 'auto'
    applyTheme(appConfig.theme)
    appConfig.llamaCppDir = config.llamaCppDir || ''
    appConfig.modelsDir = config.modelsDir || ''
    appConfig.llamaCppDownloadDir = config.llamaCppDownloadDir || ''
    appConfig.modelDownloadDir = config.modelDownloadDir || ''
    appConfig.downloadSource = config.downloadSource || 'hf'
    appConfig.language = config.language || 'auto'
    if (appConfig.language === 'auto') {
      let l: 'zh' | 'en' = config.resolvedLanguage === 'en' ? 'en' : 'zh'
      // 客户端环境兼容：在 Android 等真机 WebView 运行环境下，
      // 若系统是中文但 Go 后端因沙盒无权执行 shell 退回了 'en'，由客户端真实 navigator.language 纠正为 'zh'
      if (typeof navigator !== 'undefined' && navigator.language) {
        const clientLang = getSystemLanguageClient()
        if (clientLang === 'zh' && config.resolvedLanguage === 'en') {
          l = 'zh'
        }
      }
      appConfig.resolvedLanguage = l
    } else {
      appConfig.resolvedLanguage = config.resolvedLanguage === 'en' ? 'en' : 'zh'
    }
    // Keep the default true when the backend omits/does not return trayEnabled (matches the backend loadConfig fallback)
    appConfig.trayEnabled = config.trayEnabled !== false
    // API-route mode: default false when the backend omits the field (legacy backend)
    appConfig.apiRouteMode = config.apiRouteMode === true
    // Backend field missing / old backend not returning it → undefined → default collapsed
    // (matching the backend loadConfig preset true fallback); only an explicit false
    // (user's expanded preference) yields false; on success sync back to localStorage
    // so the "persisted backend value" and the "local UI cache" stay consistent.
    appConfig.sidebarCollapsed = config.sidebarCollapsed !== false
    // Checklist dismissal: default false when the backend omits the field
    // (legacy backend / fresh install); only an explicit true hides it.
    appConfig.onboardingDismissed = config.onboardingDismissed === true
    // Remote-chat pairing: field-by-field defaults when the backend omits the
    // key (legacy config) — disabled, no host, llama-server default port, no key.
    const rc = config.remoteChat
    appConfig.remoteChat = {
      enabled: rc?.enabled === true,
      host: typeof rc?.host === 'string' ? rc.host : '',
      port: typeof rc?.port === 'number' && rc.port > 0 ? rc.port : 8080,
      apiKey: typeof rc?.apiKey === 'string' ? rc.apiKey : '',
    }
    setLocale(appConfig.resolvedLanguage)
    localStorage.setItem(THEME_KEY, appConfig.theme)
    localStorage.setItem(SIDEBAR_KEY, appConfig.sidebarCollapsed ? '1' : '0')
  } catch {} finally {
    appConfig.loaded = true
  }
}

export async function setTheme(theme: string) {
  appConfig.theme = theme
  applyTheme(theme)
  localStorage.setItem(THEME_KEY, theme)
  try {
    // 必须向后端保存原始偏好 (auto/light/dark)，不能存 effectiveTheme
    await setThemeBackend(theme)
  } catch {}
}

/** Toggle sidebar collapsed/expanded: optimistically update local state and write
 * localStorage first; swallow backend errors without rollback — it is a pure UI
 * preference, and an immediate visual rollback would flicker; failure only affects
 * the value restored on next launch (for the rollback pattern see setTrayEnabled,
 * where rolling back causes no visual jitter).
 */
export async function setSidebarCollapsed(collapsed: boolean) {
  appConfig.sidebarCollapsed = collapsed
  localStorage.setItem(SIDEBAR_KEY, collapsed ? '1' : '0')
  try {
    await setSidebarCollapsedBackend(collapsed)
  } catch {}
}

/** Dismiss the quick-start checklist: optimistic local update, backend persist
 * swallowed on failure — a pure UI preference like the sidebar toggle; failure
 * only affects the value restored on next launch. */
export async function setOnboardingDismissed(dismissed: boolean) {
  appConfig.onboardingDismissed = dismissed
  try {
    await setOnboardingDismissedBackend(dismissed)
  } catch {}
}

/** Switch model download source ("hf" | "huggingface" | "modelscope"): update local state first; on backend failure roll back and rethrow for UI feedback. */
export async function setDownloadSource(source: string) {
  const previous = appConfig.downloadSource
  appConfig.downloadSource = source
  try {
    await setDownloadSourceBackend(source)
  } catch (e) {
    appConfig.downloadSource = previous
    throw e
  }
}

/** Switch service access scope ("local" | "lan"): optimistically update local state,
 * then fetch the latest full serverConfig from the backend, change accessMode, and save
 * it whole (avoiding clobbering other fields the user set on the same page); on backend
 * failure roll back local state and rethrow for UI feedback. saveServerConfig returns
 * void; after success the backend has persisted the derived host, no refresh needed. */
export async function setServerAccessMode(mode: string) {
  const previous = appConfig.serverAccessMode
  appConfig.serverAccessMode = mode
  try {
    const scfg = await getServerConfig()
    scfg.accessMode = mode
    await saveServerConfigBackend(scfg)
  } catch (e) {
    appConfig.serverAccessMode = previous
    throw e
  }
}

/** Set the optional llama-server API key (bearer token; empty = no authentication,
 * current default): optimistically update local state, then fetch the latest full
 * serverConfig from the backend, change apiKey, and save it whole (avoiding
 * clobbering other fields the user set on the same page); on backend failure roll
 * back local state and rethrow for UI feedback (same pattern as setServerAccessMode). */
export async function setApiKey(value: string) {
  const previous = appConfig.serverApiKey
  appConfig.serverApiKey = value
  try {
    const scfg = await getServerConfig()
    scfg.apiKey = value
    await saveServerConfigBackend(scfg)
  } catch (e) {
    appConfig.serverApiKey = previous
    throw e
  }
}

/** Save the LAN remote-chat pairing (phone → PC llama-server): optimistically
 * update local state, persist through the backend's validating saveRemoteChat
 * binding; on backend failure roll back local state and rethrow for inline UI
 * feedback (same pattern as setServerAccessMode). The backend trims host/apiKey
 * and validates the shape, so a rejected profile never reaches appConfig. */
export async function setRemoteChat(profile: RemoteChatProfile) {
  const previous = appConfig.remoteChat
  appConfig.remoteChat = { ...profile }
  try {
    const saved: RemoteChatConfig = {
      enabled: profile.enabled,
      host: profile.host,
      port: profile.port,
      apiKey: profile.apiKey,
    }
    await saveRemoteChatBackend(saved)
    // Mirror the backend normalization (TrimSpace on host/apiKey) so the
    // optimistic UI shows the exact values that were persisted.
    appConfig.remoteChat = {
      enabled: saved.enabled,
      host: saved.host.trim(),
      port: saved.port,
      apiKey: saved.apiKey.trim(),
    }
  } catch (e) {
    appConfig.remoteChat = previous
    throw e
  }
}

/** Switch UI language ("zh" | "en" | "auto"): optimistically update the preference
 * locally; on backend success refresh the locale with the returned effective
 * resolvedLanguage (for auto, resolved from the system detection) so it applies
 * immediately; on backend failure roll back the preference and rethrow for UI feedback. */
export async function setLanguage(language: string) {
  const previous = appConfig.language
  appConfig.language = language
  try {
    const resolved = await setLanguageBackend(language)
    let l: 'zh' | 'en' = resolved === 'en' ? 'en' : 'zh'
    if (language === 'auto' && typeof navigator !== 'undefined' && navigator.language) {
      const clientLang = getSystemLanguageClient()
      if (clientLang === 'zh' && resolved === 'en') {
        l = 'zh'
      }
    }
    appConfig.resolvedLanguage = l
    setLocale(l)
  } catch (e) {
    appConfig.language = previous
    throw e
  }
}

// ─── 动态响应系统环境变更 ──────────────────────────────────────────
// 1. 系统深色/浅色模式变更动态监听 (包括 Android 系统主题与桌面端)
if (typeof window !== 'undefined' && window.matchMedia) {
  const mq = window.matchMedia('(prefers-color-scheme: dark)')
  const onThemeMediaChange = () => {
    if (appConfig.theme === 'auto') {
      applyTheme('auto')
    }
  }
  if (mq.addEventListener) {
    mq.addEventListener('change', onThemeMediaChange)
  } else if ((mq as any).addListener) {
    (mq as any).addListener(onThemeMediaChange)
  }
}

// 2. 操作系统语言动态切换监听
if (typeof window !== 'undefined') {
  window.addEventListener('languagechange', () => {
    if (appConfig.language === 'auto') {
      const l = getSystemLanguageClient()
      appConfig.resolvedLanguage = l
      setLocale(l)
    }
  })
}

/** Toggle the Windows system tray switch: optimistic local update, takes effect once
 * the backend persists it (disabling removes the tray icon immediately; re-enabling
 * requires an app restart, see the comment in wails.ts).
 * On backend failure roll back local state and rethrow for UI feedback. */
export async function setTrayEnabled(enabled: boolean) {
  const previous = appConfig.trayEnabled
  appConfig.trayEnabled = enabled
  try {
    await setTrayEnabledBackend(enabled)
  } catch (e) {
    appConfig.trayEnabled = previous
    throw e
  }
}
