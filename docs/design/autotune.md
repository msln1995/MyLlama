# 设计参考：硬件感知一键调优（auto-tune）

MyLlama 的硬件感知一键调优：读 GGUF 真实指标 + 本地硬件快照，为每个模型自动规划最优 llama-server 参数。

> 权威来源：`core/autotune.go`（策略 doc-comment + 常量 + 纯函数 `tuneModelConfig`）、
> `core/benchbw.go`（实测 RAM 带宽校准）、`core/app.go`（`TuneModelConfig` 绑定）、
> `frontend/src/views/ModelSettings.vue` + `frontend/src/lib/modelTune.ts`（前端联动）。
> 本文描述**当前工作区**实现；与代码冲突时以代码为准。
>
> 注：「serving-GPU 规划目标」（`tunePlanTarget`，按服务器配置里的 DeviceID 选卡）已落地在
> `core/autotune.go`，本文按当前实现记录。

## 目标与纯函数契约

读模型 GGUF 头里的真实指标（层数、GQA/MLA KV 几何、训练上下文、MoE 专家几何），加上本地硬件
快照（GPU 厂商 / 显存、内存、物理核、实测 RAM 带宽），为单个模型计算一组最优 llama-server
参数。Sizing 核心是纯函数 **`tuneModelConfig(hw, m) → ModelConfig`：确定性、无全局、无时间**——
同样输入永远得到同样输出。

为什么坚持纯函数：每条规则都能在单测里用构造输入直接验证（不需要真机、真 GGUF 文件或注入带宽）；
所有副作用被隔离在边缘——`tuneHardware()`（经 Home 页同一条硬件检测链取快照，
`getCalibratedRAMBandwidth` 读缓存/测量）、`readGGUFModelMetrics`（文件解析）、
`buildTuneModel`（把指标折算成输入）。核心没有任何可变性可出 bug。

绑定链路（`TuneModelConfig(modelID)`）：按扫描名找模型 → 解析 GGUF → `buildTuneModel` →
`tuneModelConfig` → **`mergeTunedConfig(existing, plan)`**（只把 tuner 负责的字段
——线程 / GPU 层数 / 上下文 / Flash Attention / KV 类型 / cpu-moe / ctx-checkpoints / μBatch——
覆盖为新方案，其余用户手选字段原样保留）→ **`SaveModelConfig` 持久化**并返回 `ModelConfig`；
GGUF 不可读时打固定措辞的
`[INFO]`（`tuneFallbackLogLine`，可 grep）并走保守回退。μBatch 同时经
`tuneModel.UBatchSize` 回灌进方案，让内存预算按用户真实 ubatch 计价（与启动前的
`modelLoadBytes` 一致，否则守护会拒绝调优刚写下的方案）。前端 `ModelSettings.vue` 调
`wails.ts` 的 `tuneModelConfig`，把返回值套进表单后由用户确认保存；成功提示的插值参数由
`lib/modelTune.ts` 的 `tunedSummaryParams` 生成（`cacheTypeK` 为空串 → 展示为 "f16"，空串即
后端默认 f16，不写 cache-type 行）。

## 输入折算要点

- **GPU 规划目标**（进行中特性，`tunePlanTarget`）：默认沿用历史行为——显存取 GPU 列表最大值，
  厂商整机投票（任何名字含 nvidia 的卡或有 CUDA 驱动 → nvidia；否则 amd/radeon → amd；否则
  none）。但服务器配置里持久化的 serving-GPU 选择（`ServerConfig.DeviceID`，一个稳定的
  nvidia-smi UUID——与 llama-server 子进程经 `CUDA_VISIBLE_DEVICES` 钉死的是同一个值）若能
  **精确匹配**（区分大小写）到某张探测到的卡，则由**那张卡**决定显存与厂商，使规划目标与
  llama-server 实际使用的 GPU 在多卡主机上保持一致；匹配卡的 vendor 由它自己的名字决定
  （名字无法分类时保留整机投票结果）。规划目标经与 Home 页相同的检测链取数，匹配时打
  `[INFO] tune: planning against GPU …`。
- **专家/稠密拆分**：优先走张量表逐张量求和（`readGGUFTensorSplit`，重读同一文件的张量区，
  `ggmlBlockTable` 给出各量化块布局），仅当其总和与文件大小差 ≤ `tuneTensorSplitTol`(5%) 才采纳；
  否则退回元数据估算 `estimateExpertBytes`（`bpw × MoE层数 × (expert_count+shared) × 3 × ffn ×
  embedding`，钳制到文件大小）；再不行按全 dense 处理。`isExpertTensor` 按张量名判类
  （`_exps` / `ffn_exp` / `.exp`+`ffn`），router（`ffn_*_inp`）与共享专家（`ffn_*_shexp`）算稠密，
  与 llama-server `--cpu-moe` 的显存驻留一致。
- **KV 折算（混合注意力一致性）**：`tuneModelConfig` 用 `每层KV × Layers` 算每 token 总量、
  `每层KV × ctx` 算 partial offload 的每层成本。混合注意力模型（`full_attention_interval>1`，如
  qwen3.5）只有 `ceil(block_count/interval)` 层带 KV，因此 `buildTuneModel` 把每层成本按
  `kvCacheLayers/BlockCount` 折算——两个公式同时保持正确，且每层权重 `W/L` 仍用真实层数。
- **保守回退**（GGUF 不可读）：32 层 / 1024 B/层/token / ctx 32768 / 全 dense。
- **内存口径 = 可回收的「可用」，不是「未分配」**（`getFreeMemoryGB` → `darwinMemoryBytes`）：
  Linux 走 `MemAvailable`，Android 走系统上报值，macOS 早期只用 `vm_stat` 的 `Pages free`，实测
  会把机器读成已满——24 GiB M2 上跑起 RSS 3.97 GiB 的 llama-server（KV 撑起来的）时 free 只剩
  0.06 GiB，而 inactive 队列还挂着 7.14 GiB、swap 三轮不动（3.79 GiB → 3.79 GiB，page-ins 仅
  +193）：文件缓存是随要随给的，这个读数会把上下文梯子钉死在最小档（同一模型 ctx 2048 而非
  16384），并让桌面 prompt 缓存预算直接归零。现在 macOS 取 `free + max(0, inactive − swap 已用)`：
  dirty inactive 换出去要付写盘成本，且已经深陷 swap 的机器剩下的缓存值钱更少；speculative 与
  purgeable 不单独再计（它们本就是同一队列的成员，实测合计 0.38 GiB，单列就是重复记账）。结果
  永远不低于原先的 free-only 值，只可能更宽。同一探针同时供 Home 内存卡取数，两处不会漂移成两个
  定义（修后该机可用内存 3.8 → 7.3 GiB，「已用」84% → 68%）。

## 五步决策树

```
tuneHardware(GPU厂商/VRAM/RAM/物理核/实测带宽) + tuneModel(权重/层数/KV/专家字节)
      │
      ▼
[1] 无可用 GPU (vendor==none 或 VRAM < 1536MB) ──────────▶ CPU-only 方案，返回
      │ 有 GPU
      ▼
[2] full offload 试装：A=f16 / B=q8_0(仅 NVIDIA) 在 VRAM 预算内选 ctx
      ├─ 装下 且 fullCtx ≥ 8192 ─────────────────────────▶ full offload 直接胜出
      ├─ 装下 但 fullCtx < 8192（局促）且 实测带宽>0
      │    且估得 cpu-moe ≥ 3.0 t/s 且 cpu-moe 方案可行 ──▶ 翻转到 cpu-moe，返回
      │    否则 ─────────────────────────────────────────▶ full offload 胜出
      ▼
[3] 装不下：MoE 且专家字节 ≤ 可用 RAM ───────────────────▶ cpu-moe 方案，返回
      ▼
[4] 否则对 ctx∈{8192,4096,2048} 逐层算 n=floor((VRAM−computeBuf)/每层)，
    CPU 侧余量 ≤ 可用 RAM ──────────────────────────────▶ partial offload，返回
      ▼
[5] 全不行 ─────────────────────────────────────────────▶ CPU-only 方案
```

排序理由：**全 offload 永远最快**（没有任何权重走 PCIe/DRAM），所以它是默认赢家，唯一例外由
实测数据解锁（局促的大上下文专家入 RAM 计划 > 局促的全 offload——上下文对可用性影响更大）；
cpu-moe 只在专家放得进内存时成立（专家横跨两侧时 partial offload 更好）；partial offload 是
稠密模型的兜底；CPU-only 是最后保障。上下文梯子（降序）`131072/65536/32768/16384/8192/4096/2048`
按训练上下文过滤（缺失按 32768 折算；`ctxLadderFor` 对 ≤0 输入按上限处理——只有手工构造输入才会
走到），至少保留最小档，每个方案必有 ctx。

cpu-moe 方案细节：`GPULayers="all"` + `CPUMoe=true`（专家驻留系统 RAM，GPU 承担稠密权重 + KV +
计算缓冲），**`Threads=0` 省略 threads 行**——专家 GEMM 落在 CPU 时，llama-server 的自动线程
分配实测 ≥ 手动钉死物理核数（与步骤 2 的物理核规则刻意不同，两处都写明理由）。

## 常量表（含义速查）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `tuneRAMTotalCapRatio` | 0.90 | 可用 RAM = min(可用内存*, 总量×0.90)，再乘安全系数（*口径见「输入折算要点」，macOS 含可回收文件缓存） |
| `tuneRAMSafetyFactor` | 0.85 | 15% 留给 OS 与其他进程 |
| `tuneMinUsableRAM` | 1 GiB | 退化快照的兜底下限 |
| `tuneGPUActiveMinVRAMMB` | 1536 | 低于此不算可用 GPU（512MB 级共享内存 iGPU 视为无 GPU） |
| `tuneNvidiaVRAMReserveMB` | 512 | NVIDIA 显存预留（CUDA 上下文比 ROCm 小） |
| `tuneOtherVRAMReserveMB` | 700 | AMD/其他预留 |
| `tuneQ8KVSizeRatio` | 0.53125 | q8_0 KV = 8.5 bit/权重 ÷ f16 的 16 bit |
| `tuneComputeBufBaseMB` | 384 | 计算缓冲（logits + 激活工作区）基线 |
| `tuneComputeBufLargeMB` | +256 @ ctx≥16384 | 长上下文追加 |
| `tuneComputeBufHugeMB` | +512 @ ctx≥65536 | 巨上下文再追加（超线性增长） |
| `tuneHeadroomGuardRatio` | 0.15 | f16 方案剩余 VRAM <15% 且 q8_0 买得到更大 ctx → 换 q8_0 |
| `tuneHugeCtxSafetyRatio` | 0.92 | ≥65536 档必须留 8% 余量（估算误差在 128k 档被放大，实测出现过仅剩 ~2% 的边缘案例；VRAM OOM 加载失败的代价远大于少一档 ctx） |
| `tuneCtxMax` / `tuneCtxMin` | 131072 / 2048 | 上下文梯子边界 |
| `tuneFullOffloadCrampedCtx` | 8192 | full offload 最优 ctx 低于此 = 局促（8192 是聊天尚不局促的最小档） |
| `tuneCPUMoeMinTPS` | 3.0 | 交互速度下限（t/s），翻转必须达标 |
| `tuneTensorSplitTol` | 0.05 | 张量表拆分总和与文件大小的容差 |

A/B 缓存合并（`mergeCachePlans`，full offload 与 cpu-moe 共用同一份 tie-break）：默认 f16；
q8_0 仅 NVIDIA，仅在能买到**严格更大** ctx、或 f16 已紧（剩余 <15%）而 q8_0 更大时胜出；
平局归 f16。

## 线程数：性能核上限（`core/autotune.go` + `core/preset.go`）

两处使用同一条规则，覆盖两条不同的下发路径：

- **调优路径**：`tuneModelConfig` 先按物理核数取 `threads`，再在探针认出大小核分裂
  （`PerfCores < Cores`）时压到性能核数（M2 4P+4E 纯 CPU tg128：4 线程 20.76、6 线程 18.81、
  8 线程 10.07 t/s，复测 21.47 / 19.49 / 10.02）。
- **未调优路径**：`defaultModelConfig()` 的 `Threads = -1`，而 `modelPresetKV` 只在 `> 0` 时
  写 threads 行——所以安卓上没跑过一键调优的模型**不带 `--threads`**，llama.cpp 自行按核数放大
  线程池。`cpuThreadDefault` 在 `buildServerCommand` 的安卓直连分支补上同一顶上限（实际值记入
  服务日志 `direct mode: threads -1 -> N`），显式设置永远优先，统一拓扑与未知分裂仍保持原样。

手机侧证据（vivo X60 / Exynos 1080：4×A55 2.002 GHz + 4×A78 2.6/2.808 GHz，Android 13）：

| 读数 | `-t 4` | llama.cpp 自动（8 核全开） |
| --- | --- | --- |
| llama-bench tg32（Q4_K_M，反序两轮） | 7.23 / 7.36 | 4.38 / 4.95 |
| App 内解码 t/s（Q5_K_M，A/B/A 各一轮 + 首轮两次） | 3.82 / 3.46 / 3.89 | 0.35 |
| App 内 prefill t/s | 14.15 / 14.38 / 14.67 | 11.98 |
| 同一次请求（600 词 + 128 token）墙钟 | 97–103 s | 443 s |
| 线程落核（`/proc/<pid>/task/*/stat` 按 CPU 时间加权） | **100 % 落在 cpu4-7** | 8 核各占 11–15 %，全部 96–100 % 忙 |

自动档那一次小核也被推理线程占满（cpu0-3 96–98 % 忙），系统界面、守护进程和内存回收一起排队
（当时 MemFree 139 MiB、swap 已用 2.8 GB），这是 10 倍差距的机制解释，而不只是核多了浪费时间。
两条边界结论：EAS 自己会把 4 个 worker 摆到 A78 上，所以**不需要** `--cpu-range` /
`--cpu-strict`（shell cgroup 里钉大核也无收益：22.98/7.36 vs 23.12/7.27 t/s）；只跑小核则是
3.2×/2.4× 的灾难（pp 6.89/7.16、tg 2.87/3.24），而这个档位恰好对应 App 历史上观察到的
3.1–3.6 t/s。单性能核机型（1+7，质核频率高过 80 % 阈值）不按此封顶——读成 1 线程的代价远大于
省掉的小核，`cpuThreadDefault` 对 `PerfCores < 2` 一律返回 0。

## 路由模式：没调优模型的上下文兜底（`core/memfit.go` + `core/preset.go`）

调优器只覆盖跑过一键调优的模型；桌面路由模式的预设生成器（`generateModelsPresetFrom`）历史上
给没配置项的模型**不写 `ctx-size`**，把决定权整个交给 llama-server——而服务端挑的那个数既不受
本机余量约束，也不是应用能算出来的那个数。实测（M2 24 GiB，llama.cpp b10537，路由模式
`--models-max 1`，RSS 取 router + worker 之和，每行都跑过两次）：

| 模型 / 预设条目 | 服务端日志 | RSS |
| --- | --- | --- |
| Qwen3-0.6B Q4_K_M，不写 `ctx-size` | `n_ctx_slot = 40960` | 4.86 GiB |
| 同上，`ctx-size = 40960` | `n_ctx_slot = 40960` | 4.84 GiB |
| 同上，`ctx-size = 32768` | `n_ctx_slot = 32768` | 3.98 GiB |
| Qwen3-4B-Instruct-2507 Q4_K_M，不写 `ctx-size` | `n_ctx_slot = 103936` | 17,013 / 17,062 MiB |
| 同上，`ctx-size = 16384`（应用现在写出的条目） | `n_ctx_slot = 16384` | 4,832 / 4,832 MiB |

0.6B 那三行说明省略与显式写出来是同一份分配（32,768→40,960 差的 8,192 token 恰好 0.88 GiB，
即该模型 f16 KV 的 112 KiB/token）；4B 那两行才是这项修复的代价：省略时服务端取了 103,936 token
（元数据写的是 262,144，服务端自己打折，应用无从预知），进程 17.0 GiB、swap 从 3.86 涨到
6.30 GiB，而被写死的 16,384 档同一模型只要 4.83 GiB。速度侧不成结论——同一台机器上
14 token 短提示的两次配对读数解码 25.11 vs 25.77 t/s，在噪声内，只有首轮那次 21.86 t/s 像是
刚被换页拖累。

预算侧是第二重代价：`--cache-ram` 按库里最重的一次加载计价（`heaviestPlannedLoad` →
`promptCacheRAMMiB`），而这条价在兜底之前按元数据里的训练上下文算，所以一个没调优的模型单凭
自己就能把剩余压成负数，于是**整个库、每一次会话都不再缓存**——当时本机可用 7,925 MiB，
4B 按 262,144 token 计价是 40,525 MiB，预算直接落到 0；换成应用写出的 16,384 档，价是
5,965 MiB，还剩 980 MiB。

`routerUntunedCtx` 因此把这件事显式写死：本机当前余量付得起训练上下文就照原样保留，付不起就沿
`tuneCtxLadder` 往下取能付得起的最大档，直到梯子下界 `tuneCtxMin`；保留的那个值先被
`tuneCtxMax`（131072）夹一次——没调优的 256k 模型不该拿到比同一模型调优后更大的窗口，否则
「预设与调优方案不会差一档」这条就只在梯子内成立；元数据读不出或内存探针失败时落回设置页默认 4096，与本文件其余兜底一致。计价侧
`routerModelPlan` 用同一个函数，且把「配置存在但 `CtxSize <= 0`」也当未调优处理，所以预设条目与
预算价永远描述同一次加载。安卓直连模式的同一形状也按同一条收：`directModelConfig` 现在把
「配置项存在但没有上下文」（旧配置文件、手改过的 JSON）与「压根没有配置项」一样落回应用默认的 4096，
其它字段照旧保留。这一步在安卓上是**致命**而不是浪费：以前那条形状会同时骗过启动前的内存守卫
（`modelLoadBytes` 对 `CtxSize <= 0` 不计 KV）并把窗口交给服务端（=模型训练时的上下文，桌面实测过它
真会被取到：4B 拿到 103,936 token）。两侧各自都实测过（手机闪退的根因是总足迹超出系统报告的可用内存、
设备没有 swap ⇒ 低内存查杀），组合起来的这一条是按机制推出来的、本机没有复现过的具体案例，所以修的是
「守卫批准的那次加载和服务端真正跑的那次不是同一次」这个不一致本身；历史默认 4096 在 X60 上已验证可用。

## 桌面路由模式的超卖告警（`core/memfit.go`）

安卓那条「放不下就拒绝启动」在桌面复制不了：路由模式是惰性加载、多模型共用一个服务，一条付不起
的条目不该让整个服务起不来。而实测告诉我们放任不管的代价——同一台 24 GiB 的 M2，把一个模型显式
设成本机付不起的上下文，服务端**不报错**：进程 RSS 到 15.85 GiB、swap 越过 7 GiB、日志里一个字都
没有，请求只是干等，整机被拖住。于是启动时按同一套计价（`heaviestPlannedLoad`）把库里最重的一次
加载和当前余量比一次，超了就在服务日志写一行 `[WARN]`，点名模型并把两个数字都报出来；服务照常启动。

告警按 `loadsStayInHostRAM` 收口：这条价把权重和 KV 全记在主机内存上，在苹果统一内存上这是**精确**
的（实测 `-ngl 99` 下 RSS 按 f16 KV 几何逐 token 增长，112.0 KiB/token），CPU-only 机器上也成立；
但在有独立显存可卸载的机器上它是**多计**的——多计对一个只会缩小预算的数是安全方向，对一条要用户去
改设置的告警不是，所以显存达到 `tuneGPUActiveMinVRAMMB`（1536 MiB，与调优器判定「这块卡能干活」
的门槛同一个数）的独显机器直接闭嘴。GPU 判定取服务真正用的那块卡（`tunePlanTarget` + 持久化的
`DeviceID`，读缓存的探测快照，不为启动额外起探针），而且只在已经判定超卖之后才读它。

## 实测 RAM 带宽校准（`core/benchbw.go`）

动机：cpu-moe 的解码速度受 DRAM 读带宽限制，静态规则分不清 12 GB/s 单通道 DDR4 笔记本和
60 GB/s 双通道 DDR5 台式机——而这两台机器的翻转决策应当不同。设计移植自 FreeToken 的
`moe/benchbw.py`（哲学层面）。

- **基准形状**：`GOMAXPROCS` 个 worker 各读一段不相交、8 字节对齐的区域；**写 first-touch**
  （把页变私有脏页——从未写过的零页会走共享零页映射，测不出真 DRAM）；1 次未计时 warmup +
  3 次计时取**中位数**；工作集 512 MiB（封顶为空闲 RAM 的 1/4，下限 8 MiB——全 cache 工作集会
  报出虚高的缓存带宽）；展开的 uint64 读 + 8 条独立累加链保证计算不拖累到 DRAM 速度之下；
  **整数求和**避免任意内存内容引发 denormal/NaN 停顿；`benchSink` 让每次归约可被观测
  （校验和进了合理性错误消息），防死代码消除；`runtime.GC()` 先行。单位是**十进制 GB/s**
  （1e9 B/s，与内存营销口径一致，可直接对比）。
- **指纹缓存**：`llama-desktop-benchcache.json`（原子写 0644）。键 = SHA-256 前 16 hex，
  输入 `v1|gpu|vram|ramTotal|cores|logical|cpu`——**含 CPU 型号**（CPU 封装是内存子系统最能标识
  组件，只有六字段会在同板换 CPU 时碰撞）、**排除 RAMFreeGB**（每秒都变，会让缓存永久失效，而
  实测带宽取决于内存子系统而非当前占用）。版本字段 `1` 让未来格式变更干净失效。
- **single-flight**：`benchMu` 临界区横跨整个「查缓存→测量→落盘」——并发两次调优点击只测一次，
  后来者阻塞后直接命中刚写好的缓存。
- **合理性窗口 [2.0, 500.0] GB/s**：窗外值视为坏计时器/沙箱，拒绝而不翻转决策（取反形式
  `!(v>=min && v<=max)` 同时挡掉 NaN/Inf）。**失败不缓存**——瞬时抖动下次点击重试一次即可。
- **fail-open**：任何失败（文件坏、版本不符、指纹不匹配、窗口外、测量出错）→ 带宽 0 →
  `tuneModelConfig` 所有规则原样走静态行为，翻转永不触发。基准只花首次点击 ~1-2 s，**永远不
  fail 整个 tune**。
- **吞吐估算**：`estimateCPUMoeTPS = 带宽×1e9 / 每token活跃专家字节`（单位自洽：B/s ÷ B/t = t/s）；
  任一输入 ≤0 返回 0 =「无估计」。活跃字节 = `ExpertBytes × ExpertUsedFrac`（缺失按整池算——
  慢方向）。

## 已知保守性（刻意为之）

1. **共享专家不计入**：`expert_shared_count` 标记的共享专家每个 token 都被读取，但不进
   `expertBytesPerToken` → t/s 估计偏乐观；固定 3.0 t/s 下限吸收这份松弛。
2. **活跃占比缺失按整池算**：`ExpertUsedFrac` 缺失时假设整个专家池都在流式读取——只会压制翻转
   （慢方向），绝不会错误启用翻转。
3. **CUDA Blackwell 下限的保守比较**（硬件快照侧，`core/llamacpp.go` / `core/sysinfo.go`）：
   compute capability ≥ 12.0（RTX 50 系）要求 CUDA ≥ 12.8，否则二进制报 "no kernel image"；
   cudart 只能从 DLL 文件名解析主版本（如 "12"），**裸主版本无法证明 ≥ 12.8，故 12.x 一律不满足
   下限**。该结论经 `GetCUDA` 进入 `tuneHardware` 的厂商判定（`hasNVIDIA || cuda.Available`）。
4. **元数据专家估算只求量级正确**：混合量化文件（如 Q4_K_M 混 Q4_K/Q6_K）无法精确折算，
   `moeBytesPerWeight` 是粗表——精确路径是张量表，且带 5% 校验，不合格即退回或按 dense。
5. **GGUF 不可读 → 全保守回退**（32 层 / 1024 B/层/token / ctx 32768 / 全 dense），宁可给出
   能跑的保守参数，不给猜出来的激进参数。
6. **巨上下文档强制留 8% 余量**：KV/计算缓冲估算误差在 128k 档被放大（见常量表）。
7. **基准全链路 fail-open**：可疑即 0，静态规则兜底（见上）。

## 实测过但（暂）未采纳

Mac 侧：M2 + llama-b10537 + Qwen3-0.6B Q4_K_M，`-r 3`，均取 ± 区间中值；重复跑动在 1% 内。
手机侧：vivo X60（Exynos 1080）+ llama-b11671 + Qwen3-4B Q4_K_M（2.32 GiB），反序两轮。

1. **调大 `--ubatch-size` 换 prefill 速度：无收益，只有代价。**
   Metal 后端（`-ngl 99`）ubatch 512 / 1024 / 2048 / 4096 下 pp512 = 1884.80 / 1888.55 /
   1884.40 / 1887.92 t/s，pp2048 = 1535.38 / 1544.73 / 1539.53 / 1537.04 t/s——四个档位之间
   相差不到 1%，而运行开销随 ubatch 线性上涨（见常量表 512→2048 的 +518→+2130 MiB）。
   所以调优器**从不**替用户抬高 ubatch：`mergeTunedConfig` 只把用户已有的值带进预算计价，抬
   高 ubatch 的结果是用上下文换不到好处的内存。
2. **调小 ubatch 换上下文（仅 CPU-only 计划）：手机侧复测已做，收益不成立，维持默认 512。**
   同机 CPU 后端（`-ngl 0`）pp512 随 ubatch 变小反而略快：128 = 547.75、256 = 536.29、
   512 = 521.16、1024 = 514.09、2048 = 511.33 t/s，且 128 档比 512 档少占约 400 MiB。按
   4B 的 147 KiB/token 折算，这 400 MiB ≈ 2.7k token，接近一档最小阶梯。
   **vivo X60（Exynos 1080，4×A78+4×A55，`-t 4 -b 2048 -p 512 -n 32 -r 2`，两轮反序）实测
   这条曲线在纯 NEON 路径上不成立**：pp512 均值 128 = 23.05、256 = 22.83、512 = **23.49**、
   1024 = 24.05 t/s（pass A 降序 25.98/23.91/23.27/23.37，pass B 升序 22.72/22.39/23.07/22.11；
   两轮整体差 ~3%，与 cpu7 从 2808 MHz 掉到 2392 MHz 的热压同向，即档位间差异全在这份漂移里）。
   解码同样平坦（tg32 均值 7.12 / 6.98 / 6.54 / 6.91）。所以调优器**不**为 CPU-only/安卓计划
   改动 ubatch：往下调买不到速度却要白花 ~400 MiB 的预算余量，往上调（1024）读数受顺序与温度
   支配、不可复现，而 ubatch 正是启动守卫计价的那一项（peak RSS 两轮都是 4.8 GiB）。手机端
   prefill 只有 ~23 t/s 这个量级，真正的杠杆在核的选取而不是批的大小——见「线程数：性能核上限」。
3. **`modelLoadBytes` 把 KV 全额记在宿主内存上，在统一内存上是精确值而非保守猜测**：
   `-ngl 99 --cache-ram 0` 下 ctx 8192→32768 让 RSS 从 1,371 MiB 涨到 4,060 MiB，
   Δ = 112.0 KiB/token，与该模型 f16 KV 几何（112 KiB/token）逐位吻合——Apple 统一内存里
   「显存」KV 就是宿主内存。只有独显（CUDA）机器上这一项才是多记，而 prompt 缓存预算只会往下
   收，多记是安全方向。
