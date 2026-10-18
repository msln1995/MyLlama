import { describe, it, expect } from 'vitest'
import {
  estimateMemory,
  kvCacheTypeFactor,
  memorySummaryRows,
  runtimeOverheadBytes,
  KV_BYTES_PER_TOKEN_F16,
  RUNTIME_OVERHEAD_BYTES_AT_512,
  HEADROOM_TIGHT_GAP_BYTES,
  type MemoryEstimate,
  type MemorySummaryCopy,
} from '../lib/modelMemory'

const GIB = 1024 * 1024 * 1024

// Realistic baseline mirroring the mock preview data: a ~2.3 GB Qwen3-4B GGUF,
// 16 GB device with 9.4 GB free (numbers in bytes). kvBytesPerTokenF16 is 0 here
// on purpose — that is the "not scanned" value, so every assertion below that
// prices KV with KV_BYTES_PER_TOKEN_F16 is also the fallback test.
const BASE = {
  modelSizeBytes: 2_476_000_000,
  kvBytesPerTokenF16: 0,
  ctxSize: 4096,
  cacheTypeK: '',
  cacheTypeV: '',
  ubatchSize: 512,
  totalMemBytes: 16 * GIB,
  freeMemBytes: Math.floor(9.4 * GIB),
}

/** The three priced terms of BASE, with the runtime overhead at the default μBatch. */
const BASE_TOTAL = BASE.modelSizeBytes + 4096 * KV_BYTES_PER_TOKEN_F16 + RUNTIME_OVERHEAD_BYTES_AT_512

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
