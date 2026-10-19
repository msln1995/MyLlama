import { describe, it, expect } from 'vitest'
import {
  estimateMemory,
  kvCacheTypeFactor,
  checkpointRetentionBytes,
  memorySummaryRows,
  runtimeOverheadBytes,
  cpuResidentLayers,
  CTX_CHECKPOINT_CAP,
  KV_BYTES_PER_TOKEN_F16,
  RUNTIME_OVERHEAD_BYTES_AT_512,
  RUNTIME_OVERHEAD_METAL_BYTES_AT_512,
  HEADROOM_TIGHT_GAP_BYTES,
  type MemoryEstimate,
  type MemorySummaryCopy,
} from '../lib/modelMemory'

const GIB = 1024 * 1024 * 1024

// Realistic baseline mirroring the mock preview data: a ~2.3 GB Qwen3-4B GGUF,
// 16 GB device with 9.4 GB free (numbers in bytes). kvBytesPerTokenF16 is 0 here
// on purpose — that is the "not scanned" value, so every assertion below that
// prices KV with KV_BYTES_PER_TOKEN_F16 is also the fallback test. swaWindow is 0
// too: this model has no sliding window, so it checkpoints nothing.
const BASE = {
  modelSizeBytes: 2_476_000_000,
  mmprojBytes: 0,
  kvBytesPerTokenF16: 0,
  ctxSize: 4096,
  cacheTypeK: '',
  cacheTypeV: '',
  ubatchSize: 512,
  swaWindow: 0,
  ctxCheckpointsOff: false,
  isAndroid: false,
  // CPU-shaped by default: no Apple GPU serving, so every assertion below keyed
  // on RUNTIME_OVERHEAD_BYTES_AT_512 prices the reserve the phone/CPU way.
  gpuLayers: 'auto',
  // 0 = the scan published no block count; the reserve helper then assumes the
  // backend's own fallback layer count.
  modelLayers: 0,
  appleMetalServing: false,
  totalMemBytes: 16 * GIB,
  freeMemBytes: Math.floor(9.4 * GIB),
}

/** The priced terms of BASE summed (its checkpoint term is 0: no window). */
const BASE_TOTAL = BASE.modelSizeBytes + 4096 * KV_BYTES_PER_TOKEN_F16 + RUNTIME_OVERHEAD_BYTES_AT_512

/** gemma-3-1b's scanned geometry: 26 layers x 1024 B of f16 KV per token, a
    512-token sliding window — the fixture the checkpoint numbers were measured
    against (core/memfit.go's ctxCheckpointCap doc). */
const SWA_PER_TOKEN = 26 * 1024
const SWA_WINDOW = 512
const SWA_CHECKPOINTS = CTX_CHECKPOINT_CAP * SWA_PER_TOKEN * SWA_WINDOW

describe('kvCacheTypeFactor', () => {
  it('empty string and f16/bf16 share the f16 ratio of 1', () => {
    expect(kvCacheTypeFactor('')).toBe(1)
    expect(kvCacheTypeFactor('f16')).toBe(1)
    expect(kvCacheTypeFactor('bf16')).toBe(1)
  })

  it('f32 doubles, quant types take their measured footprint', () => {
    // Every quantized row is the RSS step llama.cpp actually allocated for that
    // type (core/autotune.go's kvCacheTypeSizeRatio carries the same table with
    // its measurements). The clean 0.5 this test used to assert for q8_0
    // understated the real cache by 6%, and the 0.3125 / 0.25 guesses by up to
    // 20% - a saving the page claimed and the guard did not.
    expect(kvCacheTypeFactor('f32')).toBe(2)
    expect(kvCacheTypeFactor('q8_0')).toBe(0.53125)
    expect(kvCacheTypeFactor('q5_0')).toBe(0.3441)
    expect(kvCacheTypeFactor('q5_1')).toBe(0.3754)
    expect(kvCacheTypeFactor('q4_0')).toBe(0.2816)
    expect(kvCacheTypeFactor('q4_1')).toBe(0.3129)
    expect(kvCacheTypeFactor('iq4_nl')).toBe(0.2816)
  })

  it('unknown strings degrade to the f16 ratio (never understate)', () => {
    expect(kvCacheTypeFactor('made_up')).toBe(1)
  })
})

describe('estimateMemory', () => {
  it('weights + f16 KV at the documented per-token constant + default overhead', () => {
    const est = estimateMemory(BASE)
    expect(est.weightsBytes).toBe(BASE.modelSizeBytes)
    expect(est.kvBytes).toBe(4096 * KV_BYTES_PER_TOKEN_F16)
    expect(est.overheadBytes).toBe(RUNTIME_OVERHEAD_BYTES_AT_512)
    expect(est.totalBytes).toBe(BASE_TOTAL)
    expect(est.kvSavedBytes).toBe(0)
    expect(est.availableBytes).toBe(BASE.freeMemBytes)
    expect(est.headroom).toBe('ok')
  })

  it('a vision model carries its projector in the weights term', () => {
    // The scan publishes the projector the preset loads (core/gguf.go's
    // mmprojBytes) and the guard adds the same file to the weights
    // (core/autotune.go's tuneWeightsBytes). Leaving it out priced a multimodal
    // load as text-only — by more than the whole runtime reserve.
    const MMPROJ = 638_000_000
    const est = estimateMemory({ ...BASE, mmprojBytes: MMPROJ })
    expect(est.weightsBytes).toBe(BASE.modelSizeBytes + MMPROJ)
    expect(est.totalBytes).toBe(BASE_TOTAL + MMPROJ)
    // Absent (not scanned / no projector) and impossible values add nothing.
    for (const mmproj of [0, -MMPROJ, Number.NaN]) {
      expect(estimateMemory({ ...BASE, mmprojBytes: mmproj }).weightsBytes).toBe(BASE.modelSizeBytes)
    }
  })

  it('a q8_0 pair charges the cache llama.cpp actually allocated', () => {
    // 0.53125 (17/32), the measured q8_0 footprint — the clean half this case
    // used to assert understated a 4096-token cache by 6%: 271 MiB on the
    // fallback geometry, enough for the page to call a plan healthy that the
    // start guard refuses.
    const est = estimateMemory({ ...BASE, cacheTypeK: 'q8_0', cacheTypeV: 'q8_0' })
    expect(est.kvBytes).toBe(4096 * KV_BYTES_PER_TOKEN_F16 * 0.53125)
    expect(est.kvSavedBytes).toBe(4096 * KV_BYTES_PER_TOKEN_F16 * (1 - 0.53125))
    // Savings keep the headroom verdict healthier but never invent data
    expect(est.headroom).toBe('ok')
  })

  it('mixed K/V types average the two factors', () => {
    // Mean of the halves is the measurement, not an assumption: on the M2 a
    // q8_0/f16 pair stepped 0.7656 of the f16 cache in both K/V orders, exactly
    // 49/64 — the q8_0 half reading as 0.53125/2 against a plain 0.5.
    const f16Kv = 4096 * KV_BYTES_PER_TOKEN_F16
    expect(estimateMemory({ ...BASE, cacheTypeK: 'q8_0', cacheTypeV: '' }).kvBytes).toBe(f16Kv * (49 / 64))
    expect(estimateMemory({ ...BASE, cacheTypeK: '', cacheTypeV: 'q8_0' }).kvBytes).toBe(f16Kv * (49 / 64))
    expect(estimateMemory({ ...BASE, cacheTypeK: 'q8_0', cacheTypeV: '' }).kvSavedBytes).toBe(f16Kv * (15 / 64))
  })

  it('invalid ctx drops the KV term but the weights still carry their overhead', () => {
    for (const ctx of [0, -4096, Number.NaN]) {
      const est = estimateMemory({ ...BASE, ctxSize: ctx })
      expect(est.kvBytes).toBe(0)
      expect(est.kvSavedBytes).toBe(0)
      expect(est.overheadBytes).toBe(RUNTIME_OVERHEAD_BYTES_AT_512)
      expect(est.totalBytes).toBe(BASE.modelSizeBytes + RUNTIME_OVERHEAD_BYTES_AT_512)
    }
  })

  it('unknown model size yields a KV + overhead total', () => {
    const est = estimateMemory({ ...BASE, modelSizeBytes: 0 })
    expect(est.weightsBytes).toBe(0)
    expect(est.totalBytes).toBe(4096 * KV_BYTES_PER_TOKEN_F16 + RUNTIME_OVERHEAD_BYTES_AT_512)
  })

  it('nothing known prices nothing: no weights and no ctx means no overhead either', () => {
    const est = estimateMemory({ ...BASE, modelSizeBytes: 0, ctxSize: 0 })
    expect(est.overheadBytes).toBe(0)
    expect(est.totalBytes).toBe(0)
    expect(est.headroom).toBe('unknown')
  })

  it('available memory prefers the free probe and falls back to total', () => {
    expect(estimateMemory(BASE).availableBytes).toBe(BASE.freeMemBytes)
    expect(estimateMemory({ ...BASE, freeMemBytes: 0 }).availableBytes).toBe(16 * GIB)
    expect(estimateMemory({ ...BASE, freeMemBytes: 0, totalMemBytes: 0 }).availableBytes).toBe(0)
  })

  it('headroom verdicts: over / tight / ok / unknown', () => {
    // Over: estimate exceeds available
    expect(
      estimateMemory({ ...BASE, freeMemBytes: 0, totalMemBytes: 2 * GIB }).headroom
    ).toBe('over')
    // Tight: fits but the remaining gap is below the 2 GiB threshold
    expect(
      estimateMemory({ ...BASE, freeMemBytes: 0, totalMemBytes: BASE_TOTAL + HEADROOM_TIGHT_GAP_BYTES - 1 }).headroom
    ).toBe('tight')
    // Unknown: no memory probe
    expect(estimateMemory({ ...BASE, freeMemBytes: 0, totalMemBytes: 0 }).headroom).toBe('unknown')
    // Unknown: nothing to compare (no weights, no ctx)
    expect(
      estimateMemory({ ...BASE, modelSizeBytes: 0, ctxSize: 0 }).headroom
    ).toBe('unknown')
  })

  it('a device with exactly weights + KV free is over, not tight: the guard prices overhead too', () => {
    // The regression this term exists for: with the old weights + KV estimate
    // this device had a zero gap and read "偏紧"; the backend's start guard
    // (core/memfit.go) refuses the same plan outright.
    const available = BASE.modelSizeBytes + 4096 * KV_BYTES_PER_TOKEN_F16
    const est = estimateMemory({ ...BASE, freeMemBytes: 0, totalMemBytes: available })
    expect(est.totalBytes).toBe(available + RUNTIME_OVERHEAD_BYTES_AT_512)
    expect(est.headroom).toBe('over')
  })

  it('μBatch scales the overhead term linearly, as the backend reserves it', () => {
    // The same law as core/autotune.go's runtimeOverheadBytes: the magnitude is
    // the phone's (1.19 GB beyond weights + KV at ubatch 512, rounded up), the
    // linearity is the M2 sweep (128/256/512/1024/2048 → 114/248/518/1051/2130
    // MiB, i.e. the term doubles when μBatch doubles).
    expect(estimateMemory({ ...BASE, ubatchSize: 1024 }).overheadBytes).toBe(2 * RUNTIME_OVERHEAD_BYTES_AT_512)
    expect(estimateMemory({ ...BASE, ubatchSize: 256 }).overheadBytes).toBe(RUNTIME_OVERHEAD_BYTES_AT_512 / 2)
    // A bigger μBatch can turn an otherwise comfortable device tight.
    const available = BASE_TOTAL + HEADROOM_TIGHT_GAP_BYTES + 1
    expect(estimateMemory({ ...BASE, freeMemBytes: 0, totalMemBytes: available }).headroom).toBe('ok')
    expect(
      estimateMemory({ ...BASE, ubatchSize: 2048, freeMemBytes: 0, totalMemBytes: available }).headroom
    ).toBe('over')
  })

  it('a missing μBatch charges the default 512, the llama-server default', () => {
    for (const ub of [0, -512, Number.NaN]) {
      expect(estimateMemory({ ...BASE, ubatchSize: ub }).overheadBytes).toBe(RUNTIME_OVERHEAD_BYTES_AT_512)
    }
  })

  it('a Metal-served load charges the measured floor plus the host weight copy', () => {
    // The M2 readings behind RUNTIME_OVERHEAD_METAL_BYTES_AT_512: RSS minus
    // weights minus f16 KV is +149/+186 MiB (4B Q4_K_M, ctx 8192, paired), +185
    // (same model, ctx 32768) and +160 (0.6B, ctx 8192), against the phone's
    // 1.1-1.2 GB for the same term. Same μBatch law on both shapes.
    const MIB = 1024 * 1024
    // 3600 MiB over 36 blocks prices the split at exactly 100 MiB per block.
    const mac = { ...BASE, appleMetalServing: true, modelSizeBytes: 3600 * MIB, modelLayers: 36 }
    expect(estimateMemory(mac).overheadBytes).toBe(320 * MIB)
    expect(estimateMemory({ ...mac, gpuLayers: '' }).overheadBytes).toBe(320 * MIB)
    expect(estimateMemory({ ...mac, gpuLayers: '36' }).overheadBytes).toBe(320 * MIB)
    expect(estimateMemory({ ...mac, gpuLayers: '99' }).overheadBytes).toBe(320 * MIB)
    // Every block left on the host is charged again as weight bytes: measured as
    // phys_footprint, the 36-layer 4B Q4_K_M grew its dirty memory by 608 MiB
    // with 9 blocks on the CPU, 1092 with 18 and 2401 with all of them, i.e. its
    // own per-block weight. The flat CPU figure this replaced under-priced the
    // worst case by three quarters of the model.
    expect(estimateMemory({ ...mac, gpuLayers: '27' }).overheadBytes).toBe(1220 * MIB)
    expect(estimateMemory({ ...mac, gpuLayers: '18' }).overheadBytes).toBe(2120 * MIB)
    expect(estimateMemory({ ...mac, gpuLayers: '0' }).overheadBytes).toBe(3920 * MIB)
    // Unparseable setting: priced as the worst case, every block on the host.
    expect(estimateMemory({ ...mac, gpuLayers: 'half' }).overheadBytes).toBe(3920 * MIB)
    // The μBatch scaling touches the floor, not the weight copy.
    expect(estimateMemory({ ...mac, gpuLayers: '0', ubatchSize: 1024 }).overheadBytes).toBe((640 + 3600) * MIB)
    // Only a Mac with an Apple GPU gets this reserve at all: the flag is the
    // caller's answer for darwin + vendor apple.
    expect(estimateMemory({ ...mac, gpuLayers: 'all', appleMetalServing: false }).overheadBytes).toBe(
      RUNTIME_OVERHEAD_BYTES_AT_512
    )
    // An unscanned model is priced with the backend's fallback block count (32),
    // never with a silent "unknown means the cheaper figure".
    expect(estimateMemory({ ...BASE, appleMetalServing: true, modelSizeBytes: 3600 * MIB, gpuLayers: '20' }).overheadBytes).toBe(
      (320 + 1350) * MIB
    )
    // The floor alone is real headroom: the cheaper plan is what the backend's
    // router budget hands back as prompt cache.
    const cpu = estimateMemory({ ...mac, appleMetalServing: false })
    const m = estimateMemory({ ...mac, appleMetalServing: true })
    expect(cpu.totalBytes - m.totalBytes).toBe(RUNTIME_OVERHEAD_BYTES_AT_512 - RUNTIME_OVERHEAD_METAL_BYTES_AT_512)
  })

  it('cpuResidentLayers counts the blocks a gpu-layers setting leaves on the host', () => {
    expect(cpuResidentLayers('all', 36)).toBe(0)
    expect(cpuResidentLayers('auto', 36)).toBe(0)
    expect(cpuResidentLayers('', 36)).toBe(0)
    // A number only reaches full offload when it covers the model's blocks.
    expect(cpuResidentLayers('99', 36)).toBe(0)
    expect(cpuResidentLayers('36', 36)).toBe(0)
    expect(cpuResidentLayers('+36', 36)).toBe(0)
    expect(cpuResidentLayers('35', 36)).toBe(1)
    expect(cpuResidentLayers('30', 36)).toBe(6)
    expect(cpuResidentLayers('0', 36)).toBe(36)
    // An unscanned model is priced with the backend's fallback count (32).
    expect(cpuResidentLayers('30', 0)).toBe(2)
    expect(cpuResidentLayers('32', 0)).toBe(0)
    expect(cpuResidentLayers('0', Number.NaN)).toBe(32)
    // Anything the backend's integer parse would refuse is the worst case: every
    // block on the host. Padded and fractional spellings are among them - the
    // preset writer refuses such a value outright, so this only keeps the two
    // sides pricing the same string the same way.
    expect(cpuResidentLayers('half', 36)).toBe(36)
    expect(cpuResidentLayers('-1', 36)).toBe(36)
    expect(cpuResidentLayers('12.5', 36)).toBe(36)
    expect(cpuResidentLayers(' 36 ', 36)).toBe(36)
  })

  it('a scanned model prices KV with its own GGUF geometry, not the constant', () => {
    // The measured geometries behind the binding: Qwen3-0.6B Q4_K_M caches
    // 112 KiB per token and Qwen3-4B-Instruct-2507 144 KiB, against the 128 KiB
    // fallback. Same context, different model — the old constant gave both the
    // same KV term, so the page could read "充足" for a model the start guard
    // (core/memfit.go, which reads the same geometry) refuses.
    const QWEN3_06B = 112 * 1024
    const QWEN3_4B = 144 * 1024
    expect(estimateMemory({ ...BASE, kvBytesPerTokenF16: QWEN3_06B }).kvBytes).toBe(4096 * QWEN3_06B)
    expect(estimateMemory({ ...BASE, kvBytesPerTokenF16: QWEN3_4B }).kvBytes).toBe(4096 * QWEN3_4B)
    expect(estimateMemory({ ...BASE, kvBytesPerTokenF16: QWEN3_4B }).totalBytes).toBe(
      BASE.modelSizeBytes + 4096 * QWEN3_4B + RUNTIME_OVERHEAD_BYTES_AT_512
    )
  })

  it('the geometry drives the quantized-cache savings row too', () => {
    // The measured factor scales whatever the real geometry is; pricing the
    // saving off the fallback constant while the cost came from the model would
    // invent slack.
    const hybrid = 28 * 1024 // a model caching on every 4th of its 28 layers
    const est = estimateMemory({ ...BASE, kvBytesPerTokenF16: hybrid, cacheTypeK: 'q8_0', cacheTypeV: 'q8_0' })
    expect(est.kvBytes).toBe(4096 * hybrid * 0.53125)
    expect(est.kvSavedBytes).toBe(4096 * hybrid * (1 - 0.53125))
    expect(est.kvBytes + est.kvSavedBytes).toBe(4096 * hybrid)
  })

  it('an unknown or impossible geometry falls back to the documented constant', () => {
    for (const perToken of [0, -1, Number.NaN]) {
      expect(estimateMemory({ ...BASE, kvBytesPerTokenF16: perToken }).kvBytes).toBe(4096 * KV_BYTES_PER_TOKEN_F16)
    }
  })

  it('a sliding-window model is charged the checkpoint retention the guard charges', () => {
    // 8 x (26,624 B/token x a 512-token window) — the same term
    // core/memfit.go's ctxCheckpointBytes adds to the planned footprint, so a
    // page that priced only weights + KV could call a windowed model's plan
    // healthy while the device's own memory decision refused or warned about it.
    const swa = { ...BASE, kvBytesPerTokenF16: 26_624, swaWindow: 512, ctxSize: 2048 }
    const retention = CTX_CHECKPOINT_CAP * 26_624 * 512
    const est = estimateMemory(swa)
    expect(est.checkpointBytes).toBe(retention)
    expect(est.totalBytes).toBe(BASE.modelSizeBytes + 2048 * 26_624 + retention + RUNTIME_OVERHEAD_BYTES_AT_512)
    // The saving a quantized cache brings applies to the copies as well.
    expect(
      estimateMemory({ ...swa, cacheTypeK: 'q8_0', cacheTypeV: 'q8_0' }).checkpointBytes
    ).toBe(Math.round(retention * 0.53125))
  })

  it('no window, no context, the pin or Android price no retention', () => {
    const swa = { ...BASE, kvBytesPerTokenF16: 26_624, swaWindow: 512, ctxSize: 2048 }
    const keep = estimateMemory(swa).checkpointBytes
    expect(keep).toBeGreaterThan(0)
    for (const off of [
      { swaWindow: 0 },
      { ctxSize: 0 },
      { ctxCheckpointsOff: true },
      { isAndroid: true },
    ]) {
      expect(estimateMemory({ ...swa, ...off }).checkpointBytes).toBe(0)
    }
    // A model the scan could not price carries no window, so it is charged the
    // fallback KV geometry and nothing else — as the backend's fallback does.
    expect(estimateMemory({ ...BASE, swaWindow: 0 }).checkpointBytes).toBe(0)
  })
})

describe('checkpointRetentionBytes', () => {
  const perToken = 26_624

  it('is the cap times one window-sized copy of the cache', () => {
    expect(checkpointRetentionBytes(512, perToken, 16384, 1, false, false)).toBe(CTX_CHECKPOINT_CAP * perToken * 512)
  })

  it('clamps the copy to the context when that is the smaller window', () => {
    expect(checkpointRetentionBytes(512, perToken, 256, 1, false, false)).toBe(CTX_CHECKPOINT_CAP * perToken * 256)
  })

  it('takes the cache-type factor of the KV term', () => {
    expect(checkpointRetentionBytes(512, perToken, 4096, 0.53125, false, false)).toBe(
      Math.round(CTX_CHECKPOINT_CAP * perToken * 512 * 0.53125)
    )
  })

  it('degrades to 0 for absent or impossible inputs', () => {
    for (const [swa, tokens, ctx] of [
      [0, perToken, 4096],
      [-512, perToken, 4096],
      [Number.NaN, perToken, 4096],
      [512, 0, 4096],
      [512, perToken, 0],
      [512, perToken, Number.NaN],
    ]) {
      expect(checkpointRetentionBytes(swa, tokens, ctx, 1, false, false)).toBe(0)
    }
  })
})

describe('runtimeOverheadBytes', () => {
  it('is the constant at the reference μBatch and linear through the origin', () => {
    expect(runtimeOverheadBytes(512)).toBe(RUNTIME_OVERHEAD_BYTES_AT_512)
    expect(runtimeOverheadBytes(128)).toBe(RUNTIME_OVERHEAD_BYTES_AT_512 / 4)
    expect(runtimeOverheadBytes(2048)).toBe(4 * RUNTIME_OVERHEAD_BYTES_AT_512)
    expect(runtimeOverheadBytes(1024) + runtimeOverheadBytes(2048)).toBe(6 * RUNTIME_OVERHEAD_BYTES_AT_512)
  })

  it('falls back to the default for absent or non-positive values', () => {
    for (const ub of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(runtimeOverheadBytes(ub)).toBe(RUNTIME_OVERHEAD_BYTES_AT_512)
    }
  })

  it('floors a fractional μBatch to whole tokens', () => {
    // The form field is a token count; 768.5 is charged as 768.
    expect(runtimeOverheadBytes(768.5)).toBe(Math.round((RUNTIME_OVERHEAD_BYTES_AT_512 * 768) / 512))
  })
})

describe('memorySummaryRows', () => {
  const copy: MemorySummaryCopy = {
    ctxThreads: 'Context / Threads',
    kv: 'KV cache',
    estimate: 'Weights + KV est.',
    kvSaved: 'KV saved',
    available: 'Available',
    headroom: 'Headroom',
    threadsAuto: 'auto',
    headroomOk: 'Plenty ✓',
    headroomTight: 'Tight',
    headroomOver: 'Insufficient',
  }

  it('full data renders the draft row set in order', () => {
    const est = estimateMemory({ ...BASE, cacheTypeK: 'q8_0', cacheTypeV: 'q8_0' })
    const rows = memorySummaryRows(est, { ctx: 4096, threads: 8, cacheTypeK: 'q8_0', cacheTypeV: 'q8_0' }, copy)
    expect(rows.map((r) => r.key)).toEqual(['ctxThreads', 'kv', 'estimate', 'kvSaved', 'available', 'headroom'])
    expect(rows[0].value).toBe('4096 · 8')
    expect(rows[0].mono).toBe(true)
    expect(rows[1].value).toBe('q8_0')
    expect(rows[2].value).toBe(`≈ ${(est.totalBytes / GIB).toFixed(2)} GB`)
    expect(rows[3].tone).toBe('ok')
    expect(rows[3].value).toMatch(/^−/)
    expect(rows[5].value).toBe('Plenty ✓')
    expect(rows[5].tone).toBe('ok')
    expect(rows[5].mono).toBe(false)
  })

  it('mixed KV types render as a K / V pair', () => {
    const est = estimateMemory({ ...BASE, cacheTypeK: 'q8_0' })
    const rows = memorySummaryRows(est, { ctx: 4096, threads: 8, cacheTypeK: 'q8_0', cacheTypeV: '' }, copy)
    expect(rows.find((r) => r.key === 'kv')?.value).toBe('q8_0 / f16')
  })

  it('auto threads (-1) uses the localized auto label', () => {
    const rows = memorySummaryRows(estimateMemory(BASE), { ctx: 4096, threads: -1, cacheTypeK: '', cacheTypeV: '' }, copy)
    expect(rows[0].value).toBe('4096 · auto')
  })

  it('invalid ctx shows a dash placeholder and keeps the row', () => {
    const rows = memorySummaryRows(estimateMemory(BASE), { ctx: 0, threads: 8, cacheTypeK: '', cacheTypeV: '' }, copy)
    expect(rows[0].value).toBe('— · 8')
  })

  it('unknown inputs drop their rows instead of guessing (honest data)', () => {
    const est: MemoryEstimate = {
      weightsBytes: 0,
      kvBytes: 0,
      checkpointBytes: 0,
      overheadBytes: 0,
      totalBytes: 0,
      kvSavedBytes: 0,
      availableBytes: 0,
      headroom: 'unknown',
    }
    const rows = memorySummaryRows(est, { ctx: 0, threads: -1, cacheTypeK: '', cacheTypeV: '' }, copy)
    expect(rows.map((r) => r.key)).toEqual(['ctxThreads', 'kv'])
  })
})
