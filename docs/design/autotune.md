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
| `tuneRuntimeOverheadMB` / `tuneRuntimeOverheadUBatch` | 1280 @ 512 | 既非权重也非 KV 的运行开销（批计算缓冲、图 scratch、采样器内存），随 ubatch 线性过原点。CPU-only/安卓口径：Exynos 1080 上 4.17 GB 模型在 ctx 4096/32768 的绝对残差 1.19/1.10 GB。**不适用于 Mac**：同机 `-ngl 0` 的残差是 +2504 MiB（RSS）/脏内存 +2401 MiB，比手机高一倍，因为留在主机上的权重在 Mac 上是第二份拷贝（见下一行） |
| `tuneMetalRuntimeOverheadMB` | 320 @ 512 | Mac 加载的**下限**（全卸载时的完整口径），不是它的全部。M2 实测 RSS 减权重减 f16 KV = +149/+186（4B Q4_K_M ctx 8192 配对复测）、+185（同模型 ctx 32768）、+160（0.6B ctx 8192）、无 `-ngl` 旗标时 +141（llama.cpp 在本机默认全卸载），取最大读数的 1.7 倍向上取整。层与缓存全在 GPU 侧，没有 CPU 计算图再持有批缓冲，所以它是 1280 的四分之一；ubatch 缩放沿用线性律（ubatch 2048 实测 +302，律给 1280，多算是安全方向） |
| Mac 拆分的追加项·留在主机的层（`core/memfit.go` 的 `cpuResidentLayers`） | 320 + 权重 × 主机层数 / 总层数 | 用 phys_footprint（脏、不可回收的那一半）在同一台 M2、ctx 8192 / ubatch 512 上配对实测：36 层 4B Q4_K_M（每层权重 66.1 MiB）留 9 层在主机时脏内存 +608 MiB、留 18 层 +1092、全留 +2401；28 层 0.6B Q4_K_M（每层 13.5 MiB）留 14 层 +134、全留 +371。映射文件那一段始终 ~2400 MiB 且可回收，所以这是真·第二份拷贝，量级正好等于主机上那些层的权重。`'' / auto / all` 与「数值 ≥ 模型层数」不留层在主机（只有下限）；无法解析或负数按最坏情况全留。层数来自 `ModelInfo.Layers`（`tuneLayerCount`，读不到 header 时兜底 32），前端 `cpuResidentLayers` / `reserveBytes` 是同一条规则 |
| Mac 拆分的追加项·`--cpu-moe` 搬下 GPU 的专家（`core/memfit.go` 的 `hostWeightBytes`） | 320 + 专家字节 × max(0, 钉住的层数 − 留在主机的层数) / 总层数 | 同一台 M2、OLMoE-1B-7B Q4_0（16 层、3746 MiB 权重，其中专家池 `ModelInfo.ExpertBytes` 3456 MiB）、ctx 4096 / ubatch 512 配对实测：`-ngl 99` 脏内存 575 MB，`-ngl 99 --cpu-moe` 4035 MB（重复读数相差 2 MB 以内），阶跃 3460 MiB ≈ 整个专家池；此前这一项只按 320 MiB 下限计价，等于给一次加载少算 3.4 GB。`--cpu-moe` 算全部层、`--n-cpu-moe N` 算前 N 层（按层数封顶），两个旗标同时出现按前者（更贵的那一半）；`-ngl 99 --n-cpu-moe 16` 读到 4035 MB，与 `--cpu-moe` 一致，说明两个旗标只是同一件事的不同写法。重叠方向也是实测的：`-ngl N` 卸载的是**后** N 层（主机留前 L−N 层），`--n-cpu-moe K` 钉住的是**前** K 层，所以只有越过主机切分线的那几层才是追加项——实测 `-ngl 12 --n-cpu-moe 4` 1704 MB 对 `-ngl 12` 1706 MB、`-ngl 4` 配 `--n-cpu-moe 8/12` 3505/3504 MB 对裸 `-ngl 4` 3504 MB（三种钉法都零阶跃），而 `-ngl 8 --n-cpu-moe 12` 是 3254/3253 MB，比裸 `-ngl 8` 的 2604 MB 多 650 MB = 4 层的专家，不是 8 层。两者相加正好是「主机层的 dense 权重 + 被搬下 GPU 的专家」，不会重复计价。非 Apple 平台不带这一项（手机上主机张量就是权重项已覆盖的映射页）；dense 模型专家池为 0，旗标不花钱。前端 `cpuMoeHostLayers` / `reserveBytes` 是同一条规则。速度轴也是实测否决的：同一台 M2、`llama-bench -t 8 -r 3`，`-ngl 99` 解码 94.43 ± 4.75 t/s、prefill 1065.74 ± 8.74，`-ngl 99 -ncmoe 16` 只有 28.94 ± 1.87 / 227.97 ± 6.82——CPU 读的就是 Metal 同一块 DRAM，把专家搬到 CPU 既不省内存也丢速度。因此调参在 Apple Silicon 从不发这个旗标，设置页在 macOS arm64 上换成实测警告文案（`frontend/src/lib/platform.ts` 的 `warnsMoeCpuSplit`） |
| 形状选择（`core/memfit.go` 的 `loadRuntimeOverheadBytes`） | darwin + Apple GPU | 计划、启动守卫、超卖告警与前端估算共用的判定链：只有 macOS 且服务 GPU 探到 Apple 时走「下限 + 主机上的第二份权重拷贝（上面两行）」，其余平台一律 1280 口径。安卓即使探到 apple 也不算（平台旗标先否决），Mac 上插 CUDA 卡也不算 |
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

**这两条边界结论里的第一条，后来被真机加长过（2026-10-19，见下面「熄屏就是锁频」一节的钉核臂）**：
在整段对话式负载里，不钉核的 8192 臂命中轮解码会掉到 3.83–4.03 tok/s，而**同一预算钉在 cpu4-7 上是
5.04–5.58**——也就是说「EAS 自己会把 worker 摆到 A78」在短基准里成立，在长会话里**不再是能假设的事**。
`--cpu-range 4-7 --cpu-strict 1` 因此从「无收益」改成「待判」：要先用不钉核的复跑证明塌陷会再现，
再决定是不是让应用默认下发。

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

预算侧是第二重代价：`--cache-ram` 按能同时驻留的加载计价（`planResidentSet` →
`promptCacheRAMMiB`；驻留上限取应用的「最大加载模型数」，默认 1 时那就是库里最重的一次加载），
而这条价在兜底之前按元数据里的训练上下文算，所以一个没调优的模型单凭
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
没有，请求只是干等，整机被拖住。于是启动时按同一套计价（`planResidentSet`）把可能被驻留的加载和
当前余量比一次，超了就在服务日志写一行 `[WARN]`，点名模型并把两个数字都报出来；服务照常启动。

价算的是**合计**而不是最重的一条，因为 `--models-max` 是一个个数、不是一道内存闸门。b11720 在同一台
24 GiB 的 M2 上实测（四个 id 的 `--models-preset`、`--cache-ram 0`、`--models-max 2`，依次请求
big → small-a → small-b → small-c，RSS 取所有 llama-server 进程之和）：
109 MiB（未加载）→ 3,211（big）→ 3,663（big + small-a）→ 1,013（small-a + small-b，big 已被挤出）
→ 1,010（small-b + small-c）——每个已加载模型是它自己的 worker 进程，两条加载真的同时在内存里，
下一条请求才按 LRU 挤掉一个。所以一次会话的峰值是那 N 条最贵加载的**和**（N = 应用的「最大加载模型数」），
只按最重的一条计价会在 N>1 时少记整整一条。默认 N=1 时两者相等，这条改动对默认配置是 no-op。
告警因此报合计、并单点名其中最重的模型和它自己那一份；给档位建议时先把其余驻留模型从内存里扣掉
（`freeMem - rest`），扣完不够一条档位就不给档位，改口让用户调小「最大加载模型数」——那时用户能真的
填的那个数字才是可执行的建议。

告警按 `loadsStayInHostRAM` 收口：这条价把权重和 KV 全记在主机内存上，在苹果统一内存上这是**精确**
的（实测 `-ngl 99` 下 RSS 按 f16 KV 几何逐 token 增长，112.0 KiB/token），CPU-only 机器上也成立；
但在有独立显存可卸载的机器上它是**多计**的——多计对一个只会缩小预算的数是安全方向，对一条要用户去
改设置的告警不是，所以显存达到 `tuneGPUActiveMinVRAMMB`（1536 MiB，与调优器判定「这块卡能干活」
的门槛同一个数）的独显机器直接闭嘴。GPU 判定取服务真正用的那块卡（`tunePlanTarget` + 持久化的
`DeviceID`，读缓存的探测快照，不为启动额外起探针），而且只在已经判定超卖之后才读它。

## 第二道上限：GPU 工作集（`core/memfit.go` + `core/metalpool_darwin.m`）

Mac 上「空闲内存」不是唯一的天花板。Metal 一个进程能钉住的最大工作集是设备的
`recommendedMaxWorkingSetSize`，这台 24 GiB 的 M2 实测报 **18,186 MiB**；而内存刚空出来时应用的
RAM 预算（`min(free, 0.90 total) x 0.85` 那条）是 **18,360 MiB**——比工作集还宽 174 MiB。两个数不
跟彼此走，所以一条价在 RAM 里的加载完全可以越过设备。调优器早就把工作集收进自己的预算
（`tuneHardware.MetalPoolMB`），启动路径此前没有，这一节把它补上。

越过工作集的后果**不是换页**，是静默卡死。同一台 M2 上用一个 378 MiB 的 Qwen3-0.6B Q4_K_M
`-ngl 99` 括号法实测上下文档位：

| ctx | 加载 | `/health` | 之后的请求 |
| --- | --- | --- | --- |
| 155,648 | 成功 | 通过 | 正常生成 |
| 159,744 | **不报错** | **通过** | 每个 command buffer 失败 `kIOGPUCommandBufferCallbackErrorOutOfMemory`，每个请求 HTTP 500 `Compute error`，不重载模型不恢复 |

因此启动检查现在比**两个**数，各比各的账：合计价 vs 空闲 RAM，GPU 常驻份额 vs 工作集。
份额由 `modelLoadPricing` 的第二个返回值给出——整份价减去 `hostWeightBytes`（部分卸载留在宿主
的那份权重副本），`-ngl 0` 时直接为 0：那种加载向 Metal 要的东西是零，拿工作集提醒它是双重错位。
两道都过时报 RAM 那条（问题更大，且那句话在这种机器上成立），只越过工作集时报工作集，并把后果
写进那一行——因为症状和换页完全不同。没调优模型的上下文兜底（`routerUntunedCtx`）改用
`memoryBudgetBytes` 收口后的预算；prompt 缓存预算刻意**不**收口，它保留的是宿主侧副本。

已知多计（同向，安全）：权重项不按卸载切分线拆，所以部分卸载的 `pinned` 仍记整份权重——它只会
让告警更早响、让建议档位更小，不会放行一条真越界的加载。

## 安卓真机：prompt 缓存收口的代价（`core/bridge.go` + `scripts/android_plan_ab.sh`）

`promptCacheRAMMiB` 把 `--cache-ram` 压到「这次加载之后余量的一半」，这个数在桌面是推算，在
vivo X60 曲屏版（Exynos 1080、Android 13、MemTotal 11,559,000 kB）上是量出来的。对照的是**已装的
那一版**：0.5.3（tag `fa82191`，2026-09-17）落后 main 58 个提交，没有这条收口，它自己的启动命令行
（`ps -A -o ARGS`，加载 Qwen3-4B-Q5_K_M、上下文 8192）就是

```
llama-server --host 127.0.0.1 --port 8080 --alias Qwen3-4B-Instruct -m <internal>/LLM-Models/Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q5_K_M.gguf
  --ctx-size 8192 --batch-size 2048 --ubatch-size 512 --threads 4 --ctx-checkpoints 0 --gpu-layers 0
  --cont-batching --no-webui --cache-ram 8192
```

那一刻 `llama-server` RSS 5,501 MiB、应用自身 286 MiB、WebView sandbox 46 MiB，`MemAvailable` 只剩
3,748 MiB——总内存的三分之一。这就是收口要治的那个数字。

配对复测走独立二进制的同一套参数形状（`scripts/android_plan_ab.sh`：3 个无共同前缀的 ~1,360 token
请求 + 一段逐轮增长的 4 轮对话，各轮取 64 token；同一个 4B Q4_K_M、同一个 ctx 8192，只换
`--cache-ram`；每组跑两次，因为单次读数在这台机器上没有意义——同一个 arm 复测总时长 446s vs 590s）：

| `--cache-ram` | RSS 中位数 (MiB) | RSS 峰值 (MiB) | 负载总时长 | 缓存命中轮 grow1/2/3 |
| --- | --- | --- | --- | --- |
| 8192（旧版实拿值） | 5,334 / 4,934 | 5,977 / 5,976 | 446 / 590 s | 31/30/39 · 38/39/41 s |
| 1351（这条收口对同一个形状算出的值） | 4,696 / 4,596 | 5,550 / 5,725 | 481 / 498 s | 29/30/32 · 24/26/27 s |

中位数少约 600 MiB（≈11%），而**缓存命中的那些轮一次都没有变慢**——整批最快的读数
（24/26/27s）恰恰出自收口那一组。少掉的只是那段没人复用的缓存驻留，代价落在提示词重算上，而这段
负载的重算成本本来就相同（各 `grow` 轮的前缀复用率一致）。这台机器上「留一半」因此不是保守，是白捡。

混淆项说清楚：1351 的两组跑在会话后段、机身更热，它们的 `MemAvailable` 低谷（3,945 / 3,537 MiB）
反而不如 8192 组（3,989 / 3,806）。整机余量受温控和其他进程摆布，不是可比的量；同机同负载下可比的
是那个进程自己的 RSS 中位数。

应用自己的那条链路也走了一遍：HEAD 构建（debug 签名，b11717 与旧版同一引擎）装上后，对**同一个
形状**（4B Q4_K_M、ctx 8192、threads 4、batch 2048 / ubatch 512、`--gpu-layers 0`）真的发出

```
llama-server … --ctx-size 8192 … --cache-ram 1150
```

8192 被收成了 1150（同一条 `promptCacheRAMMiB`，只是这一刻的余量比独立复测那两组更少，所以数更小）。
整段负载跑下来 RSS 只在 4,985–5,206 MiB 之间动（中位 5,109），而独立那两组 `--cache-ram 8192` 会
一路顶到 5,977——收口买到的正是这条带子的上沿。和旧版应用对表只能看方向、不能看绝对值，因为旧版当时
加载的是 Q5_K_M（权重档位比 Q4_K_M 高一截）：旧版中位 5,751 / 峰 5,864、整机余量低谷 2,553 MiB、
负载总时长 663s 且最后一轮卡到 148s；这次的中位 5,109、低谷 2,926 MiB、总时长 516s、最慢的缓存命中轮
38s。引擎同版本（b11717）、参数形状同形状，变的只有这条预算和权重档位这两项。

这两组数各自取的窗口是**负载段本身**，不是采样器看到的整条序列：这次 516s 的负载落在前 131 个采样
（≈524s，中位 5,108 / 余量低谷 2,913），旧版 663s 的负载几乎就是它那 157 个采样的全部。窗口要说清楚
是因为这次采样器在负载结束后又盯了 ~20 分钟空闲尾巴，那段里进程 RSS 会一路衰减到中位 4,431、整机余量
回到 4.4 G——把全程序列当负载段来比会得出「中位 4,7xx、低谷 2,531」这种和上面打架的数（复算时就是这么
差点把一组正确的数改错）。跨屏态也一样不可比：同一个 `--cache-ram 8192`，熄屏独立臂的负载段中位只有
3,960，亮屏那两组是 5,334 / 4,934。

顺带一句：这次装的是 gradle 直接产出的 debuggable 变体（`build/android/app/build/outputs/apk/debug/`，
与 `build/bin` 那个包同一份代码、同一把 debug 签名，只是带 `debuggable` 标志），为的是能用 `run-as`
以应用自己的 uid 去看它的私有目录和试着建外部存储路径——下面的坑就是这么现的形。

还有一个**未收口的坑**，就在这次重装里现的形：`androidModelsBase` 想把模型放到应用专属外部存储
（`/storage/emulated/0/Android/data/<pkg>/files/LLM-Models`，shell 可直接写入，几 GB 的权重不必重下），
靠的是从 JNI 的内部 files 目录**推**路径。这台 vivo / Android 13 上推出来的路径存在但应用建不了它：
装完后 `/storage/emulated/0/Android/data/<pkg>` 一直没有被创建，用 `run-as` 拿应用自己的 uid 去
`mkdir` 也是 `Permission denied`（连 `/storage/emulated` 都进不去）。于是 `os.MkdirAll` 失败，
`androidModelsBase` 按设计静默回落到内部目录——功能上没事（内部与外部是同一块物理盘），代价是
**那条外部路径从没在这类真机上验证过**，而且模型一旦落在 `/data/user/<n>/<pkg>/files` 里，卸载即删、
shell 也塞不进去，重装必然要重下一遍权重。真要吃到外部存储的便利，得走 Java 侧
`Context.getExternalFilesDir()` 让系统来建这个目录，而不是在 Go 里拼路径。

## 熄屏就是锁频：这台手机上频率才是第一位协变量（`scripts/android_plan_ab.sh`）

为了把「app 前台 / 亮屏」从 prompt 缓存的 A/B 里剥掉，跑了一组熄屏、服务停、独立 llama-server 的臂
（同一条形状，8192 与 1150 各一对，并反序复测）。结论是**这个剥离动作把被测对象换掉了**：这台 vivo 一熄屏
就把三个簇一起锁到 ~1.15 GHz。

| 状态 | cpu0 簇 | cpu4 簇 | cpu7 簇 |
| --- | --- | --- | --- |
| 熄屏待机 `scaling_max_freq` | 1,170 | 1,144 | 1,144 |
| 亮屏待机 `scaling_max_freq` | 2,002 | 2,600 | 2,808 |

亮屏那一行就是 `cpuinfo_max_freq`，也就是**亮屏时一点没压**；熄屏是大核 2.45 倍降频。触发它的是显示状态、
不是热：电池温度 31.3 °C、电量 100%、`settings get global low_power` = 0。要收回一句此前说错的话——
「热区里没有一个读数超过 35 °C」是把 `Permission denied` 的报错当成了数据，`/sys/class/thermal/*` 对 shell
根本不可读，那条观察作废。

四臂（都是负载段本身的采样，无空闲尾巴；RSS 单位 MiB）：

| 臂 | 屏态 | 预算 | 冷 prefill | 冷解码 | RSS 中位 / 峰 | 命中轮耗时 | 实测频率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K1r | 亮屏 | 8192 | 20.2–22.1 | 5.61–5.77 | 5,284 / 5,846 | 30.7 / 30.3 / 32.7s | cpu7 中位 2,080 |
| K2 | 亮屏 | 1351 | 20.4–22.7 | 5.57–5.93 | 4,921 / 5,464 | 25.5 / 26.5 / 28.5s | cpu7 中位 2,080 |
| I1 | 熄屏 | 8192 | 17.8–21.0 | 5.59–5.80 | 3,960 / 5,406 | 28.8 / 29.0 / 33.5s | 无 |
| I2 | 熄屏 | 1150 | 13.8–14.8 | 3.70–3.95 | 4,576 / 5,571 | 37.6 / 39.4 / 40.9s | 无 |
| J1 | 熄屏 | 1150 | 13.5–14.1 | 3.61–3.97 | 4,497 / 5,419 | 37.1 / 38.5 / 38.4s | 无 |
| J2 | 熄屏 | 8192 | 13.8–14.6 | 3.41–4.60 | 4,922 / 5,714 | 37.4 / 39.1 / 41.7s | 无 |

频率列从 K 组才开始采（I/J 四臂当时只记 RSS 与余量）。三点读法：

1. **之前怀疑的「后跑的臂更热所以更慢」不成立**：反序复测里先跑的 1150（J1）并没有变快，速率跟后跑的 J2
   同档。I1 是熄屏四臂里唯一落在高速率档的，孤立无对，不能作为任何结论的证据。
2. **频率的影响比这条预算大一个量级，这条有实测锚**：K2（亮屏，cpu7 中位 2,080 MHz）prefill 20.4–22.7、
   命中轮 25.5–28.5s；J2（熄屏，上限 1,144 MHz）prefill 13.8–14.6、命中轮 37.4–41.7s——1.82 倍时钟换来
   1.5 倍速率，量级完全对得上。而同一熄屏频率域里 8192 对 1150（J2 对 J1）只差 1–3s。
   I1 属于哪个时钟域是**由速率反推的**（它和 K2 同档），没有实测频率，这句只作旁证。
3. **收口省 RSS 的方向仍然成立**：同域 J1 4,497 < J2 4,922，与亮屏那组（4,696 / 4,596 对
   5,334 / 4,934，峰 5,550 / 5,725 对 5,977 / 5,976）同向；只有 I1 反着，而它正是那个孤立臂。
   钉核那两臂再补一次同向证据：L1（8192，钉大核）中位 5,278，K2（1351，不钉）4,921。

### 同频亮屏对照先给出「收口快 35–40%」，再被钉大核的臂否掉（K1r / K2 / L1 / L2）

前面所有配对都没法给速度定性，因为两臂不在同一个频率域。K1r（`--cache-ram 8192`）与 K2（1351）第一次把
这条补上：**两臂每簇 `scaling_cur_freq` 中位数完全一致（2002 / 1664 / 2080 MHz）**、同一模型同一 ctx
同一请求序列、各自一次。

| 阶段 | | 冷 prefill | 解码 tok/s | 命中轮墙钟 |
| --- | --- | --- | --- | --- |
| 阶段一（3 个互不相同的 prompt，各 128 token） | 8192 不钉（K1r） | 22.10 / 20.86 / 20.16 | 5.61 / 5.77 / 5.71 | 84.3 / 86.8 / 91.4s |
| | 1351 不钉（K2） | 22.69 / 21.66 / 21.53 | 5.62 / 5.93 / 5.57 | 82.6 / 83.9 / 87.6s |
| 阶段二（同一前缀逐轮增长，各 64 token） | 8192 不钉 | 19.82 → 16.73 / 15.66 / 15.16 | **4.49 → 3.91 / 4.03 / 3.83** | 83.3 → 30.7 / 30.3 / 32.7s |
| | 1351 不钉 | 20.35 → 17.30 / 15.89 / 15.77 | **5.81 → 5.55 / 5.24 / 4.89** | 78.3 → 25.5 / 26.5 / 28.5s |

当时读出来的是：阶段一两臂完全同档，差别从阶段二第一次请求起出现并一直保持（3.8–4.0 对 4.9–5.6），
像是收口不但省内存还买回三成速度。**这个结论在钉核臂面前没有站住**：

| 臂 | 预算 | 亲和性（toybox `taskset` 掩码） | 命中轮解码 | 命中轮墙钟 | RSS 中位 |
| --- | --- | --- | --- | --- | --- |
| L1 | 8192 | `f0` = cpu4-7（A78） | **5.58 / 5.20 / 5.04** | 25.4 / 26.4 / 28.1s | 5,278 |
| K1r | 8192 | 不钉 | 3.91 / 4.03 / 3.83 | 30.7 / 30.3 / 32.7s | 5,284 |
| K2 | 1351 | 不钉 | 5.55 / 5.24 / 4.89 | 25.5 / 26.5 / 28.5s | 4,921 |
| L2 | 8192 | `0f` = cpu0-3（A55） | 0.93（只跑到第一条请求即中止） | 372.6s（同一请求 L1 是 80.6s） | — |

L1 是**同样 8192 的大预算、同样 ~5.28 GB 的 RSS、同样 2,080 MHz 的时钟**，只要钉在性能簇上，命中轮就落在
4.89–5.55 那一档，和收口臂完全重合。所以 K1r 对 K2 的速度差**不能记在 `--cache-ram` 上**——它记的是那次
加载里线程摆到了哪些核。速度这条降级为「未复现的单次观察」，能进 UI 与代码注释的仍然只有**省内存**。

L2 把另一头钉住：小核簇上同一条请求 372.6s 对大核 80.6s（4.6 倍），解码 0.93 tok/s。这比 2026-10-18
llama-bench 钉小核那轮（pp 6.89/7.16、tg 2.87/3.24）还要差得多，差别在工况：那次只有 llama-bench 自己，
这次 4 个推理线程加整个系统（`load average` 7.6）一起挤 4 个 A55。也就是说 **0.93 不是「A55 的频率」，
而是「A55 + 抢核」**。K1r 的 3.9 夹在 0.93 与 5.6 之间，最像是**部分漂移**（一部分线程落在小核），
但这条同样没被直接证明——要证它得在未钉核的臂里同时采到每个推理线程的 CPU 时间落核。

下一步（还没做）：**不钉核复跑两对 8192 对 1351**，看阶段二的塌陷会不会再现。再现，就按落核与亲和性去查
（`--cpu-range` / `--cpu-strict` 在 Android 构建里确实存在，见下）；不再现，K1r 就只是那次的一次性状态。

测量纪律由此加两条。第一位协变量仍是**屏幕状态**：任何对比都要把每簇 `scaling_cur/max_freq` 一起记下来
（采样器已补），并且**先确认两臂每簇 `scaling_cur_freq` 中位数一致再看差值**——上面那句「同域只差 1–3s」
就是没对齐频率读出来的。第二位是**线程落在哪些核**，而且它比第一条更隐蔽：频率采到了、两臂读数一样，
落核仍然可以不同（K1r 与 K2 就是这样被误读成预算差别的）。所以真正的对照必须钉核，钉了核才知道
「不钉」那次落在哪儿。落核**不靠读 `/proc/*/stat` 的最后运行核**判断——那列对空闲 worker 一律报 0，
毫无信息量，已从采样器删掉。**钉核的坑**：Android 的 `/system/bin/taskset`（toybox）只吃十六进制掩码，
不认 util-linux 的 `-c 4-7`，用它起进程会 `Unknown option 'c'` 直接失败；正确用法是 `CPUSET=f0`（cpu4-7）
或 `CPUSET=0f`（cpu0-3），harness 会拼成 `taskset $CPUSET ./llama-server …`。

顺带把留了很久的「亮屏满频臂里 prefill 在 14 与 21 tok/s 之间来回跳」那颗钉子结掉：按**这条请求实际处理的
token 数**分组，四臂（K1r / K2 / L1 / L2）完全干净地分开，没有重叠——

| 实际 prefill 长度 | 读数（tok/s） |
| --- | --- |
| 1,350–1,392（整段冷 prompt） | 19.82–23.33 |
| 230–246（同一前缀的增量） | 15.16–17.30 |

所以那不是一个臂内部随机跳，而是**同一臂里两种批长度**，之前把它当成频率或调度现象是找错了轴。机制层面
只到「增量 prefill 每轮只有两三百 token，单 token 的固定开销摊薄不足」这一层，属推断、没单独验。

顺带一句：`llama-server --help` 在 Android 构建（b11717）里确实带着 `-Cr/--cpu-range lo-hi`、
`-C/--cpu-mask`、`--cpu-strict <0|1>` 和 `--prio N`，也就是说亲和性不需要靠外部 `taskset`，应用自己就能下发。
要不要把它变成默认，等上面那对未钉核的复跑结果再说。

另一个必须记的坑：**`adb forward` 会在长请求中途掉线**，表现为「臂失败」（K1 跑到 70s 吃到
`http.client.RemoteDisconnected`，H2 也是同一死法），但服务其实好好的。现在采样器每 4s 重新
`adb forward tcp:18080 tcp:8080`，工作脚本的 `chat()` 失败重试至多 3 次并在 work 文件写 `retry n: ExcName`。

产品侧推论：以后若做「锁屏继续生成」，不是省电换来的免费午餐，而是被系统按在 1.15 GHz 上跑，慢 2.45 倍是
设计如此；要不要为它抢频率（wakelock / 保持亮屏）得先把代价量清楚再说。

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
