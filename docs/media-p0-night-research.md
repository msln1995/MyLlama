# 媒体生成夜间研究记录（2026-11-09 凌晨）

对应：`docs/plans/2026-11-06-media-generation-plan.md` P2/P3 前置研究 + `docs/media-android-feasibility.md` 开放问题。
执行机：Mac14,7 / Apple M2 / 24 GB（与 P0 验证记录同一台）。全部数据为实测；夜间空闲时段执行，单任务串行 + 分钟级监控 + 10% 阈值自动终止（全程未触发误杀，tile-512 最低余量 13% 为真实观测）。

## 1. 上游发布资产调研（P2 引擎一键安装输入）

P0 时 GitHub API 403 未完成的事项，本次通过 release 页面核实：

- 固定 commit `5f8c19f9` **就是正式发布 tag `master-1057-5f8c19f`**（2026-11-08），引擎版本锚点可从裸 commit 升级为 release tag。
- 该 release 共 11 个资产：Windows x64（CPU / CUDA12 / ROCm / Vulkan）、Ubuntu x86_64（CPU / Vulkan / ROCm）、**Darwin arm64 一个**、源码包两个。
- **无 Android 资产**（证实可行性文档开放问题 #1：Android 必须 NDK 自建）。
- 资产命名模式规整：`sd-master-<sha>-bin-<Platform>[-backend].zip`，P2 安装器可按模式构造 URL（比 llama.cpp 的 nightly 甄别简单得多）。

### 1.1 关键陷阱：Darwin arm64 资产实为 x86_64

`sd-master-5f8c19f-bin-Darwin-macOS-26.7.2-arm64.zip`（37.9 MB）解包后 `otool` 显示 **sd-cli 与 libstable-diffusion.dylib 均为 x86_64**——在 Apple Silicon 上经 Rosetta 运行。实测 `--version` 正常（exit 0，输出 `commit 5f8c19f`）。

| 项 | 结论 |
| --- | --- |
| 可运行性 | 可运行（Rosetta），无 quarantine 属性（curl/Go http 下载不触发 Gatekeeper；应用内下载同理） |
| 包内布局 | `sd-cli` + `libstable-diffusion.dylib`（@rpath 依赖）+ `sd-server` + 启动脚本 |
| P2 决策输入 | **macOS 原生 Metal 性能必须自建**（P0 已验证自建流程）；直接用官方资产会把 Rosetta 开销带给用户。Windows/Linux x64 资产为原生架构，可直接采用 |

## 2. 视频 VAE 分块尺寸 A/B（P3 解码路径）

同参数（832×480、17 帧、10 步、seed 42）唯一变量 `--vae-tile-size`，对照 P0 run4（默认 256×256 分块）：

| 指标 | 256×256（run4） | 384×384（补充） | 512×512 |
| --- | --- | --- | --- |
| 分块网格 | — | 3×2=6 块 | 2 块 |
| 采样阶段 | 684 s | 684 s | 684 s（同） |
| VAE 解码 | 3573 s | **4016 s（最差）** | **2339 s（最快，-34.5%）** |
| 全程 generate_video | 4283 s | 4610 s | **3058 s（-28.6%）** |
| 解码期空闲内存低点 | 47–48% | 36% | **13%（约 3 GB）** |
| 峰值 RSS | 7.8 GiB | 8.4 GiB | 8.3 GiB |
| 产物 | .webm/VP8 可解码 | .webm/VP8 可解码 | .webm/VP8 可解码（ffprobe + 全流解码通过） |

结论与建议（三点曲线已测全）：

1. **权衡曲线非单调，384×384 是双重劣化点**（比 256 慢 12%、余量更低）——机理：分块越大单块越快，但网格覆盖的总像素面积（含重叠冗余）增大，384 恰好落在冗余增大而并行度未升的谷底。**tile 尺寸不能做成裸滑杆**，只能枚举实测可用的档位。
2. 可用档位仅两个：**256×256（默认安全档，余量 ~48%）** 与 **512×512（加速档，-34.5% 解码，余量 ~13%，仅机器空闲且余量 ≥10 GB 时可开）**。832×480 分辨率下 512 接近最优（2 块已接近无冗余覆盖）。
3. 根本解法仍是替换/增强解码路径（VAE tae、分帧流水、或等待上游 Metal 改进）——上游 11-08 仍在密集合入 Metal 相关提交（如 INT8 ConvRot），P3 前值得重新 pin 新版本复测。

## 3. Z-Image Turbo Q2_K 量化实测（Android 内存压缩路径）

组件身份：`leejet/Z-Image-Turbo-GGUF @ c61c0e42`，`z_image_turbo-Q2_K.gguf`，2,592,442,304 B，sha256 `a9cf1b03…8d58a`（下载后本地复算一致）。对照 P0 run1（Q4_K，同 prompt/尺寸/步数/seed）：

| 指标 | Q4_K + QwenQ4_K_M（P0 run1） | Q2_K + QwenQ4_K_M | **Q2_K + QwenQ2_K（最低档）** |
| --- | --- | --- | --- |
| 生成耗时（512×1024/8 步） | 182.1 s | 182.5 s | **176.0 s** |
| 峰值 RSS | 6.65 GiB | 5.31 GiB（-20%） | **4.84 GiB（-27%）** |
| 输出目验 | 锐利（基准） | 语义正确、构图保持，纹理画意感伪影 | 与左列相当——**编码器 Q2 无进一步可见退化，主模型 Q2 主导质量损失** |
| 产物 sha256 | `d4ecf12c…` | `80d24044…` | 见 outputs/p0-image-q2k-low.png |

组件身份（均经本地 sha256 复算）：Z-Image Q2_K `a9cf1b03…8d58a`（2,592,442,304 B）；Qwen3-4B Q2_K `f7ed2f57…7862`（1,669,500,000 B，unsloth 仓库 Q2_K 与 Q2_K_L 内容相同）。

内存阶梯实测要点：编码器 Q2 只省 ~0.5 GB RSS（权重差 0.9 GB，计算缓冲吃掉差额）——压缩阶梯的边际收益递减，Android 分级按实测值而非权重和推算。

结论与建议：

1. Q2_K **速度无损、内存 -20%、质量明显降级**——定位为"低内存档位"，不替代 Q4_K 默认档。目录（P2）可加 `q2-k` variant，UI 标注质量取舍。
2. **Android 12 GB 设备可行域确认（已实测）**：最低档（Q2 主模型 + Qwen Q2 编码器）峰值 RSS **4.84 GiB**，12 GB 设备（可用 ~7–8 GB）舒适；8 GB 设备（可用 ~4–5 GB）仍差一步——下一步压缩方向是计算缓冲（更小分辨率）而非继续降权重量化（边际收益已<0.5 GB）。
3. Metal 上生成耗时与权重带宽无关（激活主导），换量化不省时间——Android CPU 端预期不同（带宽敏感），R1 实测修正。

## 4. 对计划路线的影响

| 阶段 | 影响 |
| --- | --- |
| P2 | 一键安装可锚定 release tag `master-1057-5f8c19f`；Windows/Linux 用官方资产，**macOS 资产是 x86_64 陷阱**（自建或明示 Rosetta）；目录新增 Q2 variant + 质量标注；安装器镜像 llama.cpp 的 fetch→pickAsset→resume→extract→probe 流 |
| P3 | tile 尺寸做成可解释档位（默认 256 安全档）；pin 上游新版本复测 Metal 解码性能（上游活跃改进中） |
| R1 (Android) | 最低档（Z2+QwenQ2）实测 4.84 GiB 峰值 → 12 GB 设备可行；8 GB 设备需再压计算缓冲（小分辨率优先于再降量化）；NDK 自建 sd-cli 已确认为必经路径；真机内存/速度实测仍是准入条件 |

## 5. 证据索引

- 实验（本机 `~/myllama-media-p0/`）：`logs/tile512-monitor.log`、`logs/vid-tile512.log(-time.log)`、`outputs/p0-video-probe-tile512.webm`、`logs/tile384-monitor.log`、`logs/vid-tile384.log(-time.log)`、`outputs/p0-video-probe-tile384.webm`、`logs/img-q2k.log(-time.log)`、`outputs/p0-image-q2k.png`、`logs/img-q2k-low.log(-time.log)`、`outputs/p0-image-q2k-low.png`、`engine/sd-release/`（官方资产解包）、`logs/dl-q2.log`、`logs/dl-qwen-q2.log`
- 对照基线：`docs/media-generation-p0-validation.md` §5/§6
- 仓库内不留任何模型/二进制/大产物；Q2_K 权重在内置盘 `models-internal/`（外置盘已满），后续可清理
