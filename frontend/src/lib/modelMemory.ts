/**
 * Memory-estimate helpers for the ModelSettings tablet summary (design draft
 * frames ⑪⑫): a live "weights + KV cache + runtime overhead" estimate that
 * reacts to the form's ctx-size, KV-cache-type and μBatch fields, plus the
 * summary-row builder shared by the portrait bottom island and the landscape
 * sticky rail (same data, two placements).
 *
 * Honesty rules (mirroring the other tablet phases): every term is real data.
 * The weights term is the scanned GGUF file size and the KV term is priced with
 * the model's own per-token cache geometry, both from GetModels; only a model
 * whose header could not be read falls back to the documented typical-model
 * constant below, and every affected row is labeled as an estimate (预估/≈).
 * Rows whose inputs are unknown are dropped entirely instead of guessed.
 *
 * The three terms are the same three the backend's start guard prices
 * (core/memfit.go's modelLoadBytes), and that agreement is the point: an
 * estimate of weights + KV alone was 1.25 GiB (at the default μBatch) too
 * optimistic, so this page could call a plan "充足" while the same device's
 * start guard refused it (Android direct mode) or warned about it (desktop
 * router mode).
 */

import { formatBytes } from './format'

/**
 * Fallback KV-cache bytes per token at f16, K+V combined, for a typical modern
 * GQA model (kv dim ≈ 1024, ~32 layers → 2 × 32 × 1024 × 2 B). Scanned models
 * publish their real geometry through ModelInfo's kvBytesPerTokenF16 (the same
 * figure the backend tuner sizes its context ladder against), so this constant
 * only stands in for a model the scan could not price: measured real geometries
 * spread wide around it — 112 KiB/token for Qwen3-0.6B Q4_K_M, 144 KiB/token for
 * Qwen3-4B-Instruct-2507, and 28 KiB/token for a hybrid-attention model that
 * caches on every 4th layer.
 */
export const KV_BYTES_PER_TOKEN_F16 = 128 * 1024

/**
 * Runtime reserve (everything a load keeps resident that is neither weights
 * nor KV: batch compute buffers, graph scratch, sampler arenas) at the app's
 * default μBatch of 512 tokens — the frontend mirror of core/autotune.go's
 * `tuneRuntimeOverheadMB` / `tuneRuntimeOverheadUBatch`. Measured on the
 * Exynos 1080 phone (4.17 GB model at ctx 4096 and ctx 32768: 1.19 / 1.10 GB
 * beyond weights + f16 KV, so the term is context-independent) and rounded up
 * here; an M2 sweep of the same term confirmed it grows linearly with μBatch
 * (128/256/512/1024/2048 → +114/248/518/1051/2130 MiB). Keep it in sync with
 * the backend constant.
 */
export const RUNTIME_OVERHEAD_BYTES_AT_512 = 1280 * 1024 * 1024

/** The μBatch the overhead constant is priced at (also llama-server's default). */
export const OVERHEAD_REFERENCE_UBATCH = 512

/** Gap (bytes) below which the remaining headroom is flagged "tight". */
export const HEADROOM_TIGHT_GAP_BYTES = 2 * 1024 * 1024 * 1024

/** Headroom verdict of the estimate against the available memory. */
export type MemoryHeadroom = 'ok' | 'tight' | 'over' | 'unknown'

/** Inputs for the estimate; 0 always means "unknown" for byte fields. */
export interface MemoryEstimateInput {
  /** Scanned GGUF file size in bytes (0 = model not found in the scan). */
  modelSizeBytes: number
  /**
   * Model's real f16 KV bytes per token from the scan (0 = unknown, which
   * falls back to {@link KV_BYTES_PER_TOKEN_F16}).
   */
  kvBytesPerTokenF16: number
  /** Configured context size in tokens (<= 0 / NaN = not usable yet). */
  ctxSize: number
  /** Configured KV cache K type ('' = backend default f16). */
  cacheTypeK: string
  /** Configured KV cache V type ('' = backend default f16). */
  cacheTypeV: string
  /** Configured μBatch size (<= 0 / NaN = the 512 default llama-server uses). */
  ubatchSize: number
  /** Total system memory in bytes (0 = probe unavailable). */
  totalMemBytes: number
  /** Free system memory in bytes (0 = probe unavailable). */
  freeMemBytes: number
}

export interface MemoryEstimate {
  /** Weights term: the GGUF file size (0 when the model size is unknown). */
  weightsBytes: number
  /** KV-cache term, ctx-scaled and quantization-scaled (0 when ctx invalid). */
  kvBytes: number
  /** Runtime reserve term, μBatch-scaled (0 only when nothing else is known). */
  overheadBytes: number
  /** weightsBytes + kvBytes + overheadBytes (0 only when all terms are unknown). */
  totalBytes: number
  /** KV bytes saved vs an all-f16 baseline by quantized cache types (>= 0). */
  kvSavedBytes: number
  /** Free memory when known, else total memory (0 = probe unavailable). */
  availableBytes: number
  /** Headroom verdict; 'unknown' when either side of the comparison is 0. */
  headroom: MemoryHeadroom
}

/**
 * Bytes-per-element ratio of a KV cache type relative to f16 (2 bytes):
 * empty string is the backend default f16; unknown strings degrade to the
 * f16 ratio (1) so a typo can never understate the estimate.
 *
 * These are measurements, not format arithmetic, and they are the frontend
 * mirror of core/autotune.go's kvCacheTypeSizeRatio — measured as the steady
 * state RSS step from ctx 2048 to ctx 16384 on the M2 with llama.cpp b10537
 * (Qwen3-0.6B Q4_K_M, -ngl 0, twice per type, readings agreeing within 0.02%):
 * q8_0 0.5314, q5_0 0.3440, q5_1 0.3753, q4_0 0.2816, q4_1 0.3128, iq4_nl
 * 0.2815, bf16 1.0000, f32 2.0000 of the model's 114,746 B/token f16 geometry.
 * Before this table the page charged q8_0 a clean half (understating it by 6%)
 * and every other quantized type below its real size, while the backend guard
 * charged all of them the full f16 cache — two wrong numbers that disagreed.
 * Keep the two tables in sync.
 */
export function kvCacheTypeFactor(cacheType: string): number {
  switch (cacheType) {
    case '':
    case 'f16':
    case 'bf16':
      return 1
    case 'f32':
      return 2
    case 'q8_0':
      return 0.53125
    case 'q5_0':
      return 0.3441
    case 'q5_1':
      return 0.3754
    case 'q4_0':
    case 'iq4_nl':
      return 0.2816
    case 'q4_1':
      return 0.3129
    default:
      return 1
  }
}

/**
 * Runtime reserve for a given μBatch size, in bytes: the constant is priced at
 * the app's default 512 and the term is linear through the origin (see
 * {@link RUNTIME_OVERHEAD_BYTES_AT_512} for the measurements). A missing or
 * non-positive μBatch charges the default, which is also what llama-server
 * runs when the form has no value yet.
 */
export function runtimeOverheadBytes(ubatchSize: number): number {
  const ubatch = Number.isFinite(ubatchSize) && ubatchSize > 0 ? Math.floor(ubatchSize) : OVERHEAD_REFERENCE_UBATCH
  return Math.round((RUNTIME_OVERHEAD_BYTES_AT_512 * ubatch) / OVERHEAD_REFERENCE_UBATCH)
}

/**
 * f16 KV bytes per token used for the estimate: the model's real GGUF geometry
 * when the scan published it, else the documented typical-model fallback. A
 * negative or non-finite value is "unknown" too — the backend guards the same
 * field the same way (core/autotune.go's tuneKVBytesPerTokenF16), and 0 from the
 * binding means the model was never scanned, not that its cache is free.
 */
export function kvBytesPerToken(kvBytesPerTokenF16: number): number {
  return Number.isFinite(kvBytesPerTokenF16) && kvBytesPerTokenF16 > 0 ? kvBytesPerTokenF16 : KV_BYTES_PER_TOKEN_F16
}

/**
 * Compute the live memory estimate from the current form values and probes.
 * Pure: no reactive state, safe to call from computed properties and tests.
 */
export function estimateMemory(input: MemoryEstimateInput): MemoryEstimate {
  const ctx = Number.isFinite(input.ctxSize) && input.ctxSize > 0 ? Math.floor(input.ctxSize) : 0
  const weightsBytes = input.modelSizeBytes > 0 ? input.modelSizeBytes : 0
  // The per-token figure covers K+V at f16; mixed types scale each half.
  const perToken = kvBytesPerToken(input.kvBytesPerTokenF16)
  const kvAvgFactor = (kvCacheTypeFactor(input.cacheTypeK) + kvCacheTypeFactor(input.cacheTypeV)) / 2
  const kvBytes = ctx * perToken * kvAvgFactor
  const kvSavedBytes = ctx > 0 && kvAvgFactor < 1 ? ctx * perToken * (1 - kvAvgFactor) : 0
  // The reserve exists only for a load that has some price on it: with neither
  // weights nor KV known, showing a bare overhead figure would be a number
  // without a subject, and the rows keyed off the total drop instead.
  const overheadBytes = weightsBytes > 0 || kvBytes > 0 ? runtimeOverheadBytes(input.ubatchSize) : 0
  const totalBytes = weightsBytes + kvBytes + overheadBytes
  const availableBytes = input.freeMemBytes > 0 ? input.freeMemBytes : input.totalMemBytes > 0 ? input.totalMemBytes : 0
  let headroom: MemoryHeadroom = 'unknown'
  if (availableBytes > 0 && totalBytes > 0) {
    if (totalBytes > availableBytes) headroom = 'over'
    else if (availableBytes - totalBytes < HEADROOM_TIGHT_GAP_BYTES) headroom = 'tight'
    else headroom = 'ok'
  }
  return { weightsBytes, kvBytes, overheadBytes, totalBytes, kvSavedBytes, availableBytes, headroom }
}

/** Localized labels for the summary rows (resolved once by the component). */
export interface MemorySummaryCopy {
  ctxThreads: string
  kv: string
  estimate: string
  kvSaved: string
  available: string
  headroom: string
  threadsAuto: string
  headroomOk: string
  headroomTight: string
  headroomOver: string
}

export interface MemorySummaryRow {
  /** Stable row key (vue :key). */
  key: 'ctxThreads' | 'kv' | 'estimate' | 'kvSaved' | 'available' | 'headroom'
  label: string
  value: string
  /** Tone drives the value color: plain / ok (green) / warn (amber) / error. */
  tone: 'plain' | 'ok' | 'warn' | 'error'
  /** Monospaced value (numeric rows per the draft's .mono). */
  mono: boolean
}

/**
 * Build the draft's "参数速览" rows from the estimate and the raw form values.
 * The rows are shared verbatim by the portrait island and the landscape rail.
 * Rows with unknown inputs are omitted: nothing priced (no model size and no
 * ctx) → no estimate row, no memory probe → no available/headroom rows,
 * all-f16 caches → no savings row.
 */
export function memorySummaryRows(
  estimate: MemoryEstimate,
  form: { ctx: number; threads: number; cacheTypeK: string; cacheTypeV: string },
  copy: MemorySummaryCopy
): MemorySummaryRow[] {
  const rows: MemorySummaryRow[] = []
  const ctx = Number.isFinite(form.ctx) && form.ctx > 0 ? Math.floor(form.ctx) : 0
  const threads = form.threads > 0 ? String(Math.floor(form.threads)) : copy.threadsAuto
  rows.push({
    key: 'ctxThreads',
    label: copy.ctxThreads,
    value: `${ctx > 0 ? String(ctx) : '—'} · ${threads}`,
    tone: 'plain',
    mono: true,
  })
  const k = form.cacheTypeK || 'f16'
  const v = form.cacheTypeV || 'f16'
  rows.push({ key: 'kv', label: copy.kv, value: k === v ? k : `${k} / ${v}`, tone: 'plain', mono: true })
  if (estimate.totalBytes > 0) {
    rows.push({
      key: 'estimate',
      label: copy.estimate,
      value: `≈ ${formatBytes(estimate.totalBytes)}`,
      tone: 'plain',
      mono: true,
    })
  }
  if (estimate.kvSavedBytes > 0) {
    rows.push({
      key: 'kvSaved',
      label: copy.kvSaved,
      value: `−${formatBytes(estimate.kvSavedBytes)}`,
      tone: 'ok',
      mono: true,
    })
  }
  if (estimate.availableBytes > 0) {
    rows.push({ key: 'available', label: copy.available, value: formatBytes(estimate.availableBytes), tone: 'plain', mono: true })
  }
  if (estimate.headroom === 'ok') {
    rows.push({ key: 'headroom', label: copy.headroom, value: copy.headroomOk, tone: 'ok', mono: false })
  } else if (estimate.headroom === 'tight') {
    rows.push({ key: 'headroom', label: copy.headroom, value: copy.headroomTight, tone: 'warn', mono: false })
  } else if (estimate.headroom === 'over') {
    rows.push({ key: 'headroom', label: copy.headroom, value: copy.headroomOver, tone: 'error', mono: false })
  }
  return rows
}
