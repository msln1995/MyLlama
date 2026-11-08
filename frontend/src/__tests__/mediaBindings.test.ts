// Media binding wrapper tests: the wails.ts media wrappers normalize backend
// results (nil → empty list) and pass requests through untouched. The wails
// module is mocked (side-effect-bearing bridge), the wrappers themselves are
// the unit under test.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mediaMocks = vi.hoisted(() => ({
  GetMediaRuntime: vi.fn(),
  ListMediaModels: vi.fn(),
  StartMediaJob: vi.fn(),
  GetMediaJobs: vi.fn(),
  CancelMediaJob: vi.fn(),
  ListMediaArtifacts: vi.fn(),
  ListMediaInstalls: vi.fn(),
  GetMediaHTTPStatus: vi.fn(),
  SetMediaEngineDir: vi.fn(),
  InspectMediaImport: vi.fn(),
  ImportMediaComponents: vi.fn(),
  RemoveMediaInstall: vi.fn(),
}))

// Mock the generated bindings module (the side-effect-bearing Wails bridge);
// the pure pass-through/normalization wrappers in ../wails run for real.
vi.mock('../../bindings/github.com/CodeNeow/MyLlama/core/app', () => mediaMocks)
vi.mock('../../bindings/github.com/CodeNeow/MyLlama/core/models', () => ({}))

import {
  getMediaHTTPStatus,
  getMediaJobs,
  getMediaRuntime,
  listMediaArtifacts,
  listMediaInstalls,
  listMediaModels,
  setMediaEngineDir,
  startMediaJob,
} from '../wails'

beforeEach(() => {
  vi.clearAllMocks()
})

describe('media binding wrappers', () => {
  it('normalizes nil model lists to empty arrays', async () => {
    mediaMocks.ListMediaModels.mockResolvedValue(null)
    await expect(listMediaModels()).resolves.toEqual([])
  })

  it('normalizes nil job and artifact lists to empty arrays', async () => {
    mediaMocks.GetMediaJobs.mockResolvedValue(null)
    mediaMocks.ListMediaArtifacts.mockResolvedValue(null)
    await expect(getMediaJobs()).resolves.toEqual([])
    await expect(listMediaArtifacts()).resolves.toEqual([])
  })

  it('normalizes nil installs to the empty shape', async () => {
    mediaMocks.ListMediaInstalls.mockResolvedValue(null)
    await expect(listMediaInstalls()).resolves.toEqual({ installs: [], statuses: [] })
  })

  it('passes the job request through untouched and returns the snapshot', async () => {
    const req = { requestId: 'r1', modelId: 'z-image-turbo', variantId: 'q4-k', params: { prompt: 'cat', seed: 42, width: 512, height: 512, steps: 8, cfgScale: 1.0 } }
    const snap = { id: 'job-1', status: 'queued' }
    mediaMocks.StartMediaJob.mockResolvedValue(snap)
    await expect(startMediaJob(req as never)).resolves.toBe(snap)
    expect(mediaMocks.StartMediaJob).toHaveBeenCalledWith(req)
  })

  it('surfaces the engine status and http status verbatim', async () => {
    const engine = { installed: false, engineDir: '/x', enginePath: '/x/sd-cli', version: '', commit: '', backend: 'unknown', error: 'sd engine not found' }
    mediaMocks.GetMediaRuntime.mockResolvedValue(engine)
    await expect(getMediaRuntime()).resolves.toBe(engine)
    const http = { running: true, addr: '127.0.0.1:1901', state: 'running' }
    mediaMocks.GetMediaHTTPStatus.mockResolvedValue(http)
    await expect(getMediaHTTPStatus()).resolves.toBe(http)
  })

  it('forwards the engine dir setting', async () => {
    mediaMocks.SetMediaEngineDir.mockResolvedValue(undefined)
    await expect(setMediaEngineDir('/custom/engine')).resolves.toBeUndefined()
    expect(mediaMocks.SetMediaEngineDir).toHaveBeenCalledWith('/custom/engine')
  })

  it('getMediaHTTPStatus reflects a failed bind as not running', async () => {
    mediaMocks.GetMediaHTTPStatus.mockResolvedValue({ running: false, addr: '127.0.0.1:1901', state: 'failed', error: 'address already in use' })
    const st = await getMediaHTTPStatus()
    expect(st.running).toBe(false)
    expect(st.state).toBe('failed')
  })

  afterEach(() => {
    vi.clearAllMocks()
  })
})
