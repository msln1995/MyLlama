import { describe, it, expect } from 'vitest'
import {
  RECOMMENDED_MODELS,
  getDefaultFilter,
  filterRecommendedModels,
  type RecommendedModel
} from '../lib/recommendedModels'

describe('RECOMMENDED_MODELS curated catalog', () => {
  it('contains at least 4 well-tested models with unique IDs', () => {
    expect(RECOMMENDED_MODELS.length).toBeGreaterThanOrEqual(4)
    const ids = RECOMMENDED_MODELS.map(m => m.modelId)
    expect(new Set(ids).size).toBe(ids.length)
  })

  it('each model has complete metadata, reasons, and valid quant size', () => {
    for (const m of RECOMMENDED_MODELS) {
      expect(m.name.length).toBeGreaterThan(0)
      expect(m.badgeZh.length).toBeGreaterThan(0)
      expect(m.badgeEn.length).toBeGreaterThan(0)
      expect(m.speedZh.length).toBeGreaterThan(0)
      expect(m.quant).toBe('Q4_K_M')
      expect(m.fileSize).toMatch(/GB$/)
      expect(m.ramRequired.length).toBeGreaterThan(0)
      expect(m.license.length).toBeGreaterThan(0)
      expect(m.tagsZh.length).toBeGreaterThanOrEqual(2)
      expect(m.tagsEn.length).toBeGreaterThanOrEqual(2)
      expect(m.reasonZh.length).toBeGreaterThan(10)
      expect(m.reasonEn.length).toBeGreaterThan(10)
      expect(m.recommendedReasonZhDetails.length).toBeGreaterThanOrEqual(3)
      expect(m.recommendedReasonEnDetails.length).toBeGreaterThanOrEqual(3)
    }
  })

  it('has Qwen2.5-3B featured as the top recommended mobile model', () => {
    const qwen3b = RECOMMENDED_MODELS.find(m => m.modelId === 'Qwen/Qwen2.5-3B-Instruct-GGUF')
    expect(qwen3b).toBeDefined()
    expect(qwen3b?.featured).toBe(true)
    expect(qwen3b?.tagsZh).toContain('手机端首选')
  })
})

describe('getDefaultFilter', () => {
  it('returns mobile on mobile devices and all on desktop', () => {
    expect(getDefaultFilter(true)).toBe('mobile')
    expect(getDefaultFilter(false)).toBe('all')
  })
})

describe('filterRecommendedModels', () => {
  it('returns all models when filter is all', () => {
    const result = filterRecommendedModels(RECOMMENDED_MODELS, 'all')
    expect(result.length).toBe(RECOMMENDED_MODELS.length)
  })

  it('returns mobile-compatible models when filter is mobile', () => {
    const result = filterRecommendedModels(RECOMMENDED_MODELS, 'mobile')
    expect(result.length).toBeGreaterThanOrEqual(2)
    for (const m of result) {
      expect(m.category === 'mobile' || m.targetTier === 'mobile' || m.targetTier === 'all').toBe(true)
    }
  })

  it('returns desktop models when filter is desktop', () => {
    const result = filterRecommendedModels(RECOMMENDED_MODELS, 'desktop')
    expect(result.length).toBeGreaterThanOrEqual(2)
    for (const m of result) {
      expect(m.category === 'desktop' || m.targetTier === 'desktop').toBe(true)
    }
  })

  it('returns coder models when filter is code', () => {
    const result = filterRecommendedModels(RECOMMENDED_MODELS, 'code')
    expect(result.length).toBe(1)
    expect(result[0].modelId).toBe('Qwen/Qwen2.5-Coder-7B-Instruct-GGUF')
  })
})
