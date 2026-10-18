// @vitest-environment happy-dom
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import Api from '../views/Api.vue'
import { getServerLogsSince, startServer } from '../wails'

// The backend refuses to spawn a load that cannot fit in memory and returns
// that explanation as the start error. It must stay visible on the page: the
// 1s log poll replaces the console wholesale on a retention gap (and the phone
// tier clips long lines at the right edge), so a log-pushed error reads as
// "the tap did nothing".
const GUARD_MSG =
  'model Qwen3 4B Instruct needs about 8.4 GB for this load (weights + context cache + runtime overhead) but only 7.5 GB of memory is available right now'

vi.mock('../wails', () => ({
  getServerStatus: vi.fn(() => Promise.resolve({ running: false })),
  getModels: vi.fn(() => Promise.resolve([])),
  getLoadedModels: vi.fn(() => Promise.resolve([])),
  getMonitorStatus: vi.fn(() => Promise.resolve({ tps: 0, uptime: 0 })),
  getServerConfig: vi.fn(() =>
    Promise.resolve({ host: '127.0.0.1', port: 8080, maxModels: 1, cacheRam: 8192 }),
  ),
  getServerLogsSince: vi.fn(() => Promise.resolve({ entries: [], next: 0 })),
  refreshModels: vi.fn(() => Promise.resolve()),
  saveServerConfig: vi.fn(() => Promise.resolve()),
  startServer: vi.fn(() => Promise.resolve()),
  startServerWithModel: vi.fn(() => Promise.resolve()),
  stopServer: vi.fn(() => Promise.resolve()),
  unloadModel: vi.fn(() => Promise.resolve()),
}))

vi.mock('vue-router', async () => {
  const actual = await vi.importActual<typeof import('vue-router')>('vue-router')
  return { ...actual, useRouter: () => ({ push: vi.fn(), replace: vi.fn(), back: vi.fn() }) }
})

beforeEach(() => {
  localStorage.clear()
  vi.clearAllMocks()
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

async function mountApi() {
  const wrapper = mount(Api)
  await flushPromises()
  return wrapper
}

describe('Api.vue start failure banner', () => {
  it('shows the memory-fit refusal in a persistent banner', async () => {
    vi.mocked(startServer).mockRejectedValueOnce(new Error(GUARD_MSG))
    const wrapper = await mountApi()

    await wrapper.find('button.primary-btn.start').trigger('click')
    await flushPromises()
    await vi.advanceTimersByTimeAsync(600)

    const banner = wrapper.find('.action-error')
    expect(banner.exists()).toBe(true)
    expect(banner.text()).toContain(GUARD_MSG)
    wrapper.unmount()
  })

  it('keeps the banner after a retention-gap log reset replaces the console', async () => {
    vi.mocked(startServer).mockRejectedValueOnce(new Error(GUARD_MSG))
    // First poll since cursor 0 reports a gap; the subsequent full fetch
    // (since 0) replaces serverLog — exactly what used to wipe a pushed error.
    vi.mocked(getServerLogsSince).mockImplementation((since: number) =>
      Promise.resolve(
        since === 0 ? { entries: [], next: 0 } : { entries: [{ seq: 5000, text: 'evicted' }], next: 5000 },
      ),
    )
    const wrapper = await mountApi()

    await wrapper.find('button.primary-btn.start').trigger('click')
    await flushPromises()
    await vi.advanceTimersByTimeAsync(2500)

    expect(wrapper.find('.action-error').text()).toContain(GUARD_MSG)
    wrapper.unmount()
  })

  it('clears the banner when the next start attempt succeeds', async () => {
    vi.mocked(startServer).mockRejectedValueOnce(new Error(GUARD_MSG))
    const wrapper = await mountApi()

    await wrapper.find('button.primary-btn.start').trigger('click')
    await flushPromises()
    expect(wrapper.find('.action-error').exists()).toBe(true)

    await wrapper.find('button.primary-btn.start').trigger('click')
    await flushPromises()
    await vi.advanceTimersByTimeAsync(600)
    expect(wrapper.find('.action-error').exists()).toBe(false)
    wrapper.unmount()
  })
})
