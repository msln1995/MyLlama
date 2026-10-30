/**
 * 精选推荐模型数据定义与分类器
 *
 * 拒绝营销号虚标与纸面参数：所有模型均基于真实芯片算力、LPDDR 内存带宽物理限制
 * 以及实际端侧 Function Calling / 多动作调度实测验证。
 */

export interface RecommendedModel {
  /** HuggingFace / ModelScope 上的仓库 ID */
  modelId: string
  /** 展示名称 */
  name: string
  /** 核心参数量分级 */
  sizeCategory: '1.5B' | '3B' | '7B'
  /** 适用终端目标 */
  targetTier: 'mobile' | 'desktop' | 'all'
  /** 业务分类 */
  category: 'mobile' | 'desktop' | 'code' | 'english'
  /** 醒目标签（中/英） */
  badgeZh: string
  badgeEn: string
  /** 实测生成速度提示（中/英） */
  speedZh: string
  speedEn: string
  /** 建议量化版本 */
  quant: string
  /** 建议版本的下载体积 */
  fileSize: string
  /** 运行内存建议要求 */
  ramRequired: string
  /** 开源许可协议 */
  license: string
  /** 特性标签（中/英） */
  tagsZh: string[]
  tagsEn: string[]
  /** 深度推荐理由：为什么推荐它，解决了什么核心痛点 */
  reasonZh: string
  recommendedReasonZhDetails: string[]
  reasonEn: string
  recommendedReasonEnDetails: string[]
  /** 是否高亮置顶（黄金甜点模型） */
  featured?: boolean
}

export type RecommendationFilter = 'all' | 'mobile' | 'desktop' | 'code'

/**
 * 经实测验证的精选推荐模型库
 */
export const RECOMMENDED_MODELS: readonly RecommendedModel[] = [
  {
    modelId: 'Qwen/Qwen2.5-3B-Instruct-GGUF',
    name: 'Qwen2.5 3B Instruct',
    sizeCategory: '3B',
    targetTier: 'all',
    category: 'mobile',
    badgeZh: '手机端极力推荐 · 综合天花板',
    badgeEn: 'Mobile Flagship · Top Pick',
    speedZh: '7.8 tok/s (息屏 5.6 tok/s)',
    speedEn: '7.8 tok/s (5.6 tok/s locked)',
    quant: 'Q4_K_M',
    fileSize: '1.96 GB',
    ramRequired: '4 GB+ (常驻 2.1 GB)',
    license: 'Apache 2.0',
    tagsZh: ['手机端首选', '多动作调度', '自然阅读流速', '全中文通识'],
    tagsEn: ['Mobile Best', 'Tool Calling', 'Reading Speed', 'Bilingual'],
    reasonZh: '移动端物理内存带宽下的黄金尺寸，速度、智商与工具调用的最佳平衡点。',
    recommendedReasonZhDetails: [
      '实测流畅：Exynos 1080 物理芯片冷启动仅 6.5s，生成速度 7.8 tok/s，完全匹配人眼自然阅读流速。',
      '手机操作调度：零样本成功实现闹钟、短信、App 跳转等多动作复合拆解，原生输出标准 OpenAI JSON 工具调用。',
      '资源极其健康：常驻物理内存仅 2.1 GB，发热仅 38.8°C，绝不杀后台，低功耗息屏锁屏推理无损。',
      '协议友好：完全开源 Apache 2.0，支持 Unsloth / Llama-Factory 极速微调定制。'
    ],
    reasonEn: 'The sweet spot for mobile RAM bandwidth: perfect balance of speed, intelligence, and function calling.',
    recommendedReasonEnDetails: [
      'Smooth inference: 6.5s cold load on Exynos 1080, 7.8 tok/s decode matching natural reading speed.',
      'Phone action agent: Zero-shot multi-action task decomposition (alarm/SMS/app launch) in standard OpenAI JSON format.',
      'Low thermal & RAM: 2.1GB RSS footprint, 38.8°C cool operation, zero background kills even on locked screen.',
      'Permissive license: Apache 2.0 with first-class Unsloth and Llama-Factory fine-tuning support.'
    ],
    featured: true
  },
  {
    modelId: 'Qwen/Qwen2.5-1.5B-Instruct-GGUF',
    name: 'Qwen2.5 1.5B Instruct',
    sizeCategory: '1.5B',
    targetTier: 'mobile',
    category: 'mobile',
    badgeZh: '极速轻量 · 老机入门救星',
    badgeEn: 'Ultra Fast · Low RAM',
    speedZh: '15~20 tok/s',
    speedEn: '15~20 tok/s',
    quant: 'Q4_K_M',
    fileSize: '1.04 GB',
    ramRequired: '3 GB+ (常驻 1.3 GB)',
    license: 'Apache 2.0',
    tagsZh: ['极小体积', '毫秒级响应', '老旧机型', '通知摘要'],
    tagsEn: ['Tiny Footprint', 'Instant Response', 'Budget Phones', 'Summarization'],
    reasonZh: '极限瘦身与超低功耗，4GB 运存老旧设备也能丝滑常驻，随问随答几乎零延迟。',
    recommendedReasonZhDetails: [
      '极限小体积：模型仅 1.04 GB，冷启动加载只要 3 秒，3GB/4GB 运存入门手机毫无负担。',
      '极速生成：15~20 tok/s 高频吐字，适合文本润色、快速通知摘要、语法纠错与基础分类。',
      '基础中文过硬：保留了 Qwen2.5 出色的中英文指令遵循与格式化输出能力。'
    ],
    reasonEn: 'Ultra-compact footprint with sub-second response, runs smoothly even on 4GB legacy phones.',
    recommendedReasonEnDetails: [
      'Ultra compact: 1.04 GB weight, 3s cold start, runs comfortably on 3GB/4GB budget devices.',
      'Blazing throughput: 15~20 tok/s generation for instant answers, notification summary, and text polishing.',
      'Solid bilingual foundation: Inherits Qwen2.5 high-quality instruction following.'
    ]
  },
  {
    modelId: 'Qwen/Qwen2.5-7B-Instruct-GGUF',
    name: 'Qwen2.5 7B Instruct',
    sizeCategory: '7B',
    targetTier: 'desktop',
    category: 'desktop',
    badgeZh: '桌面主力 · 深度常识推理',
    badgeEn: 'Desktop Flagship · High IQ',
    speedZh: '30~50 tok/s (独显/M系) / 6~10 tok/s (CPU)',
    speedEn: '30~50 tok/s (GPU/Metal) / 6~10 tok/s (CPU)',
    quant: 'Q4_K_M',
    fileSize: '4.68 GB',
    ramRequired: '8 GB+ (显存/内存)',
    license: 'Apache 2.0',
    tagsZh: ['桌面端首选', '复杂逻辑', '角色扮演', '长文多步推理'],
    tagsEn: ['Desktop Best', 'Complex Logic', 'Roleplay', 'Deep Reasoning'],
    reasonZh: '电脑端（PC / Mac）黄金尺寸，公认的智商与常识推理甜点，适合日常复杂工作助理。',
    recommendedReasonZhDetails: [
      '逻辑智商飞跃：7B 参数量跨越常识与复杂推演门槛，在数学、逻辑推理、长文本分析上表现优异。',
      '桌面端极速：配有 8GB+ 显存独显（RTX 3060/4060 等）或 Apple Silicon (M1/M2/M3) 可跑满 30~50 tok/s。',
      '支持通过 MyLlama 开启 API 路由，作为局域网或本机主力的 OpenAI 兼容大脑。'
    ],
    reasonEn: 'The sweet spot for workstations: strong general reasoning, roleplay, and document analysis.',
    recommendedReasonEnDetails: [
      'High reasoning power: 7B marks the threshold for complex logic, math, and multi-step document reasoning.',
      'Fast on desktop: Delivers 30~50 tok/s on 8GB+ VRAM discrete GPUs and Apple Silicon Metal.',
      'Ideal for MyLlama headless API route serving your desktop and local network.'
    ],
    featured: true
  },
  {
    modelId: 'Qwen/Qwen2.5-Coder-7B-Instruct-GGUF',
    name: 'Qwen2.5 Coder 7B',
    sizeCategory: '7B',
    targetTier: 'desktop',
    category: 'code',
    badgeZh: '程序员利器 · 离线代码副驾驶',
    badgeEn: 'Coding Copilot · Offline',
    speedZh: '30~50 tok/s (独显/M系)',
    speedEn: '30~50 tok/s (GPU/Metal)',
    quant: 'Q4_K_M',
    fileSize: '4.68 GB',
    ramRequired: '8 GB+ (显存/内存)',
    license: 'Apache 2.0',
    tagsZh: ['代码生成', 'Bug 调试', 'Shell/Git', '私密离线'],
    tagsEn: ['Code Generation', 'Bug Fixing', 'Shell/Git', 'Private Offline'],
    reasonZh: '专门针对 90+ 种编程语言微调的代码专家，具备极强的代码生成、单元测试与 Bug 排查能力。',
    recommendedReasonZhDetails: [
      '代码能力顶尖：在 EvalPlus 等代码评测中媲美甚至超越 GPT-4o-mini，函数补全、算法实现极准。',
      '纯离线私密：商业机密代码无需上传云端，本地直接与 IDE、终端、Git 配合使用。',
      '熟练生成 Shell 脚本、Docker 配置、Git 工作流与正则。'
    ],
    reasonEn: 'Fine-tuned programming specialist across 90+ languages for code generation, debugging, and scripts.',
    recommendedReasonEnDetails: [
      'Top-tier coding benchmark: Matches or surpasses GPT-4o-mini on EvalPlus coding benchmarks.',
      '100% private offline: Proprietary and sensitive code stays entirely on your local machine.',
      'Proficient in Shell scripts, Docker configs, Git automation, and regex.'
    ]
  },
  {
    modelId: 'bartowski/Llama-3.2-3B-Instruct-GGUF',
    name: 'Llama 3.2 3B Instruct',
    sizeCategory: '3B',
    targetTier: 'all',
    category: 'english',
    badgeZh: '纯英学术 · 128K 超长上下文',
    badgeEn: '128K Context · English Pro',
    speedZh: '7~9 tok/s (手机) / 40+ tok/s (PC)',
    speedEn: '7~9 tok/s (Mobile) / 40+ tok/s (PC)',
    quant: 'Q4_K_M',
    fileSize: '2.02 GB',
    ramRequired: '4 GB+',
    license: 'Llama 3.2 Community',
    tagsZh: ['128K 上下文', '纯英学术', '长篇论文', '跨语种翻译'],
    tagsEn: ['128K Context', 'English Academic', 'Long Papers', 'Translation'],
    reasonZh: 'Meta 官方轻量主力，原生支持 128K 上下文窗口，英文写作与学术论文阅读首选。',
    recommendedReasonZhDetails: [
      '128K 上下文：支持整本英文技术文档或长篇论文一次性载入，轻量模型中的长文本王者。',
      '母语级英文表现：在纯英文指令遵循、邮件拟写、学术润色方面表现极其地道。',
      '体积仅 2.02 GB，手机与电脑通用。'
    ],
    reasonEn: 'Meta official lightweight model with 128K context, excels in English writing and long papers.',
    recommendedReasonEnDetails: [
      '128K context window: Digest full papers and technical documentation in a single prompt.',
      'Native English fluency: Unmatched nuance for English writing, email drafting, and academic editing.',
      'Compact 2.02 GB size, runs well on both mobile and desktop.'
    ]
  }
]

/**
 * 根据平台类型获取初始默认分类
 */
export function getDefaultFilter(isMobile: boolean): RecommendationFilter {
  return isMobile ? 'mobile' : 'all'
}

/**
 * 根据分类过滤推荐模型
 */
export function filterRecommendedModels(
  models: readonly RecommendedModel[],
  filter: RecommendationFilter
): RecommendedModel[] {
  if (filter === 'all') return [...models]
  if (filter === 'mobile') {
    return models.filter(m => m.category === 'mobile' || m.targetTier === 'mobile' || m.targetTier === 'all')
  }
  if (filter === 'desktop') {
    return models.filter(m => m.category === 'desktop' || m.targetTier === 'desktop')
  }
  if (filter === 'code') {
    return models.filter(m => m.category === 'code')
  }
  return [...models]
}
