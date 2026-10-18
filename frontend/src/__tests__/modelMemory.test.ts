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
// 16 GB device with 9.4 GB free (numbers in bytes).
const BASE = {
  modelSizeBytes: 2_476_000_000,
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

  it('f32 doubles, quant types scale down', () => {
    expect(kvCacheTypeFactor('f32')).toBe(2)
    expect(kvCacheTypeFactor('q8_0')).toBe(0.5)
    expect(kvCacheTypeFactor('q5_0')).toBe(0.3125)
    expect(kvCacheTypeFactor('q4_0')).toBe(0.25)
    expect(kvCacheTypeFactor('iq4_nl')).toBe(0.25)
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

  it('quantized KV types halve the KV term and report the saving', () => {
    const est = estimateMemory({ ...BASE, cacheTypeK: 'q8_0', cacheTypeV: 'q8_0' })
    expect(est.kvBytes).toBe(4096 * KV_BYTES_PER_TOKEN_F16 * 0.5)
    expect(est.kvSavedBytes).toBe(4096 * KV_BYTES_PER_TOKEN_F16 * 0.5)
    // Savings keep the headroom verdict healthier but never invent data
    expect(est.headroom).toBe('ok')
  })

  it('mixed K/V types average the two factors', () => {
    const f16Both = estimateMemory(BASE)
    const mixed = estimateMemory({ ...BASE, cacheTypeK: 'q8_0', cacheTypeV: '' })
    expect(mixed.kvBytes).toBeCloseTo(f16Both.kvBytes * 0.75, 6)
    expect(mixed.kvSavedBytes).toBeCloseTo(f16Both.kvBytes * 0.25, 6)
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
