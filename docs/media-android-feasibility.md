# 媒体生成 Android 可行性研究

日期：2026-11-08（Asia/Shanghai）
输入：`docs/media-generation-p0-validation.md`（macOS P0 实测）+ `docs/plans/2026-11-06-media-generation-plan.md` §3.1（Android 首版不开放本地媒体执行，需单独研究打包、进程、内存与后台限制）
性质：只读研究，无代码变更。所有"现有架构"结论均给出 `file:line` 证据；所有"预期"均标注未验证。

## 1. 结论摘要

1. **工程打包路径成立**：sd 引擎在 Android 的落地可以完全复用现有 llama.cpp 运行时的分发与执行模式（内部存储可执行文件 + 外部存储权重 + 前台服务保活），没有发现架构级阻塞。
2. **图片生成：有条件可行**。P0 实测图片峰值 RSS 6.65 GiB，8 GB 设备不可行，12 GB 设备边缘；需要量化降级/分辨率压缩后再实测。
3. **视频生成：本机数据已否定 v1 可行性**。M2 GPU 上 17 帧/10 步需 71 分钟且 VAE 解码存在内存悬崖；手机 CPU 预期再慢一个数量级以上，无实测价值，维持计划"视频不对 Android 承诺"。
4. **后续最大不确定项**：sd.cpp 的 Android arm64 官方预编译资产是否存在（上游 GitHub API 本次未确认，同计划 P0 遇到的 403）；若无则需自建 NDK 产物并纳入发布流程。

## 2. 现有 Android 执行架构（证据）

| 机制 | 证据 | 对媒体生成的意义 |
| --- | --- | --- |
| targetSdk 28 刻意保持 W^X 豁免，app 可从可写目录 exec 子进程 | `build/android/app/build.gradle:36-49`（注释明确"targetSdk 28 keeps the W^X exemption for app-writable dirs"，赋值在 :49；:120 处 exec 语境再次引用该决策），sideload-only 定位 | sd-cli 可执行文件可与 llama-server 同方式执行，无需改 targetSdk |
| 引擎放内部存储，SELinux 禁止外部挂载 exec | `core/paths.go:24-25` | sd 引擎二进制必须同样进内部 files 目录，不能放外部存储 |
| Android 上 llama.cpp 运行时为 CPU-only | `core/llamacpp.go:64`（"android = CPU-only"） | sd 引擎首版同样只做 CPU 后端；Metal/CUDA 不存在，OpenCL/Vulkan 后续单独评估 |
| 子进程共享库同目录分发 + 运行时 LD_LIBRARY_PATH 注入 | `core/llamacpp.go:178-200`（`androidLdEnv`）、`:269-270` | sd 引擎如带 ggml 共享库，沿用同一布局与环境注入 |
| 权重大文件放 app 外部存储 | `core/paths.go:22-23`（androidExternalBase） | 媒体权重（数 GB）放外部存储，与聊天模型一致 |
| 前台服务保活与服务生命周期绑定 | `AndroidManifest.xml:21-22,87-90`（FOREGROUND_SERVICE_DATA_SYNC + WailsForegroundService）、`core/bridge.go:311-339`、`core/foreground.go` | 媒体生成任务复用同一 keepAwake 接线，通知文案新增媒体语义 |
| Go 侧以 c-shared 注入（libwails.so），JNI 入口注册 | `Taskfile.yml:157-222`（NDK clang、aarch64-linux-android21、jniLibs）、`main_android.go` | 若选择 sd 进程内方案（见 §3），构建矩阵需加一个 NDK 目标 |

## 3. sd 引擎打包：两条路径

| 路径 | 说明 | 评估 |
| --- | --- | --- |
| A. 独立 `sd-cli` 可执行文件（推荐） | 与 llama-server 完全同构：NDK 交叉编译或上游预编译资产 → 运行时下载安装到内部 files 目录 → `runCmd`/`hideWindow`（Android 为 no-op）启动子进程 → 复用 `logfmt` 日志管线 | 零新进程模型；任务取消/超时/退出回收语义与计划 §5.4 完全一致；与现有 P1 runner（media_runner）天然同构 |
| B. 进程内 c-shared（libsd.so + JNI/Go cgo） | sd.cpp 编译为共享库由 libwails.so 链接或 dlopen | 省一次进程地址空间（对低内存设备有利），但引入 cgo/JNI 边界、取消语义复杂化（线程内中断 vs 杀进程）、与 P1 runner 设计冲突。不建议首版采用 |

结论：**路径 A**。理由：与已验证的 llama-server 模式零偏差；P0 已证明进程级取消/回收语义干净（SIGTERM 0s 退出、无孤儿、无假产物）。

## 4. 内存预算对照（P0 实测 → Android 设备分级）

P0 图片组合（Z-Image Turbo Q4_K 3.86 GB + Qwen3-4B Q4_K_M 2.50 GB + ae VAE 0.34 GB）+ 运行缓冲 = 峰值 RSS 6.65 GiB（M2，Metal 后端，-H 1024 -W 512）。

| 设备档位 | 可用 RAM（扣除系统） | 评估 |
| --- | --- | --- |
| 8 GB | ~4–5 GB | 不可行（权重即 6.7 GB） |
| 12 GB | ~7–8 GB | 边缘：需降低缓冲（更小分辨率/步数）或更低量化文本编码器；必须真机实测 |
| 16 GB+ | ~10 GB+ | 预计可运行（未验证） |

压缩路径（按优先级）：主模型 Q2_K（2.59 GB，上游有现成量化，质量下降需目验）→ 文本编码器 Q2（Qwen3-4B 2bit 约 1.6 GB）→ 分辨率降至 512×512 级别。组合后权重可压至 ~4.5 GB，12 GB 设备可行域。以上均为推算，**未实测**。

视频：P0 实测采样期 RSS ~7 GB + VAE 不 tiled 内存悬崖 +7 GB / tiled 59.5 min——手机 CPU 后端两项都不可接受。**结论：视频不对 Android 开放**（与计划 §3.1 一致），除非未来出现大幅压缩的 VAE 路径。

## 5. 进程、后台与系统限制

- **前台保活**：生成期间复用 `core/foreground.go` 的 dataSync 前台服务（现有 server 已如此）。注意 Android 14+ 对 `dataSync` 类型有每日时限（约 6 小时）——长生成任务需评估改用 `specialUse` 或 `mediaProcessing`（API 35+ 新增类型，正好匹配媒体生成语义），列入开放问题。
- **Doze/App Standby**：前台服务期间不适用；任务结束释放保活（现有 `keepAwakeOffFn` 语义一致）。
- **LMK/进程死亡**：Go 主进程被杀则 sd-cli 子进程成孤儿——llama-server 现有退出回收逻辑（wait goroutine）不覆盖主进程死亡场景；媒体任务恢复语义（计划 §5.4 interrupted 状态）在 Android 上同样适用，重启后发现孤儿引擎进程需按进程身份清理，不能按名杀。
- **16 KB page size**：Android 15 设备要求 NDK 产物 16KB 对齐；sd 引擎自建产物需 `-Wl,-z,max-page-size=16384` 并真机验证（本项目现有 libwails.so 是否已对齐属既有问题，不在本文范围）。
- **存储**：图片产物单张 <2 MB，作品目录放外部存储无压力；模型权重下载沿用现有任务队列。

## 6. 性能预期（全部未验证）

- M2 Metal：512×1024/8 步 = 182 s。Android CPU（8 核 arm64）ggml 对扩散模型预期慢 3–10 倍 → 单图 10–30 分钟量级，仅"能出图"，谈不上体验；OpenCL（Adreno）后端理论上可压缩到分钟级，但 sd.cpp 的 OpenCL 支持面向 Adreno 且需 NDK 特殊构建，属独立验证批次。
- 由此推导的产品建议：Android 首个媒体批次只承诺"小尺寸低步数图片生成"，UI 明示耗时预期，默认参数由后端按设备内存分级给出。

## 7. 建议路线（并入计划节奏）

| 批次 | 内容 | 前置 |
| --- | --- | --- |
| R0 = 计划 P1/P2 | 桌面媒体底座与图片闭环（Android 不在范围内，维持计划 §3.1） | P0 已完成（本文档 + p0 验证记录） |
| R1 | Android 图片实验：NDK 自建/获取 sd-cli arm64（CPU）→ 12 GB 真机跑通 512×512 出图 → 按实测决定参数分级与开放范围 | R0 的 runner/arbiter 落地（复用同一任务核心）；真机到位 |
| R2 | Android 体验化：OpenCL/Vulkan 后端评估、参数分级 UI、作品页适配（safe-area/软键盘已有基建） | R1 实测数据 |
| 不承诺 | Android 视频生成 | VAE 解码路径在桌面端先解决（P3 门槛） |

## 8. 开放问题

1. 上游是否发布 Android arm64 预编译 sd 资产（GitHub API rate limit 未确认；无则自建并纳入发布流程，含 16KB 对齐与 License 清单）。
2. `mediaProcessing` 前台服务类型（API 35）与现有 `dataSync` 的迁移/兼容策略。
3. Qwen3-4B Q2 量化 + Z-Image Q2_K 的质量目验（M2 上可先行，验证后作为 Android 参数分级的输入）。
4. 孤儿引擎进程的跨重启清理（进程身份锚点可复用 `proctime_*` 的 processStartTime 思路，但 Android 无 /proc stat 读取限制问题——`procfile_android.go` 已有进程内读取通道）。
5. 真机清单与最低开放线（建议首测：12 GB 旗舰一台 + 8 GB 中端一台各一，验证 §4 分级）。
