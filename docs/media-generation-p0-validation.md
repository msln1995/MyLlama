# 媒体生成 P0 验证记录（macOS Apple Silicon）

对应计划：`docs/plans/2026-11-06-media-generation-plan.md` 第 7 节 P0。
执行日期：2026-11-08（Asia/Shanghai）。执行机：本报告第 1 节所述单台 MacBook。
状态：已完成（P0 全项）。本文为 P0 真机验证记录；所有数字均为本机实测，不外推。

## 1. 硬件与磁盘盘点（P0-1）

| 项 | 实测值 |
| --- | --- |
| 机型 / 芯片 | Mac14,7 / Apple M2（8 CPU 核） |
| GPU | M2 集成 10 核，Metal 4 |
| 统一内存 | 24 GiB（25,769,803,776 B） |
| 操作系统 | macOS 27.3 (26C5035r) |
| 磁盘可用（实验开始时） | 约 23 GiB（APFS 容器 460Gi，已用 95%） |
| 工具链 | Xcode 27.2（Apple clang），cmake + ninja（Homebrew），aria2c、ffmpeg/ffprobe、python3+PIL |
| 可复用本地权重 | 无（仓库无 GGUF/safetensors；全部组件需下载） |
| 网络 | github.com / huggingface.co / hf-mirror.com 均可达；hf-mirror 单流约 1 MiB/s，aria2 16 连接峰值约 17 MiB/s，持续传输约 3–6 MiB/s |

结论：磁盘是最紧约束（23 GiB），本次实验组件总下载约 13.5 GiB + 引擎构建约 1 GiB，留有约 8 GiB 余量给输出与临时文件；生成输出统一写入仓库外目录 `~/myllama-media-p0/outputs/`（模型、二进制、大产物不入 Git）。

## 2. 引擎固定（P0-2）

- 引擎源：`https://github.com/leejet/stable-diffusion.cpp`
- 固定 commit：`5f8c19f98ddb84e301bb8725b5260ec79e768e22`（2026-11-08，"perf: reduce fused compute buffers (#2290)"；无对应发布 tag，上游 tag 形如 `master-<sha>` 且未覆盖 HEAD）
- 子模块固定：`ggml@3fc478e6`、`thirdparty/cutlass@ad7b2f5e`、`thirdparty/libwebm@5bf12267`、`thirdparty/libwebp@0c9546f7`、`examples/server/frontend@fa93429c`
- 构建参数：`cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSD_METAL=ON -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++`，随后 `cmake --build build --target sd-cli`
- 构建注意：全量构建会触发 `examples/server` 的 vite 前端构建，本机在 ServBay pnpm 环境下因 `~/.Trash` 扫描 EPERM 失败；只影响 server 示例，不影响 `sd-cli`（用 `--target sd-cli` 绕过）
- 产物：`build/bin/sd-cli`（约 45 MB，arm64）；`sd-cli --version` 输出 `stable-diffusion.cpp version unknown, commit 5f8c19f`
- 构建耗时：约 6 分钟（8 核并发，与模型下载并行时）

## 3. 参数基线（P0-3，`sd-cli --help` 实测）

完整输出：实验目录 `logs/sd-cli-help.txt`（307 行）。关键确认：

- 运行模式 `-M` 支持 `[img_gen, adetailer, vid_gen, vid_upscale, upscale, convert, ...]`
- 组件参数：`--diffusion-model`、`--high-noise-diffusion-model`、`--llm`（Qwen 系文本编码器）、`--t5xxl`、`--vae`、`--vae-format auto|flux|sd3|flux2|wan`、`--llm_vision`
- 视频参数：`--video-frames`（默认 1）、`--flow-shift`（默认 auto）、`-M vid_gen` 必须显式指定（与上游 wan.md 一致）
- 资源参数：`--offload-to-cpu`（权重驻留 RAM 按需进显存）、`--vae-tiling`、`--backend <component>=cpu` 形式
- 采样：`--sampling-method [euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, ...]`

## 4. 首套图片模型组件固定（P0-4）

Z-Image Turbo 量化选择依据：磁盘余量优先，选 Q4_K（上游文档示例本身使用更低量化 Q3_K，证明可运行；质量对比图显示 Q4_K 可用）。全部组件经 hf-mirror 下载，身份锚点如下（HF LFS 元数据 sha256，下载完成后本地复算比对）：

| 组件 | 来源仓库 @ revision | 文件 | 大小 | sha256 |
| --- | --- | --- | --- | --- |
| 主扩散模型 | `leejet/Z-Image-Turbo-GGUF` @ `c61c0e42` | `z_image_turbo-Q4_K.gguf` | 3,864,250,304 B | `14b375ab4f226bc5378f68f37e899ef3c2242b8541e61e2bc1aff40976086fbd` |
| 文本编码器 | `unsloth/Qwen3-4B-Instruct-2507-GGUF` @ `a06e946b` | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2,497,281,120 B | `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597` |
| VAE | `Comfy-Org/Z-Image-Turbo` @ `6fc90a3b` | `split_files/vae/ae.safetensors` | 335,304,388 B | `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38` |

注意：上游 z_image.md 指向的 VAE 来源 `black-forest-labs/FLUX.1-schnell` 现为门控仓库（HTTP 401），改用同文件的 Comfy-Org 重打包副本（内容以 sha256 锚定，与目录内 `ae.safetensors` 一致角色）。本地校验结果见第 5 节。

视频探针组件（P0-6 用）：

| 组件 | 来源仓库 @ revision | 文件 | 大小 | sha256 |
| --- | --- | --- | --- | --- |
| 主扩散模型 | `Comfy-Org/Wan_2.1_ComfyUI_repackaged` @ `123acf1c` | `split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors` | 2,838,303,560 B | `be531024cd9018cb5b48c40cfbb6a6191645b1c792eb8bf4f8c1c6e10f924dc5` |
| VAE | 同上 @ `123acf1c` | `split_files/vae/wan_2.1_vae.safetensors` | 253,815,318 B | `2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b` |
| 文本编码器 | `city96/umt5-xxl-encoder-gguf` @ `b535255b` | `umt5-xxl-encoder-Q4_K_M.gguf` | 3,655,145,312 B | `17cf97a5bbbc60a646d6105b832b6f657ce904a8a1ad970e4b59df0c67584a40` |

许可说明：Z-Image / Qwen3-4B / Wan2.1 / umt5 均为 Apache-2.0 系开放权重（以各仓库 LICENSE 为准，后续目录审核阶段二次确认）；FLUX ae.sft 随 Z-Image-Turbo 重打包分发，许可继承上游 FLUX.1（非商用限制仅针对 FLUX 主模型生成图像的条款按原始仓库说明执行，本实验仅用于本地验证）。

## 5. 图片生成验证（P0-5）

### 5.1 命令（run1 基线）

```sh
cd ~/myllama-media-p0
engine/sd.cpp/build/bin/sd-cli \
  --diffusion-model models/z_image_turbo-Q4_K.gguf \
  --vae models/ae.safetensors \
  --llm models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -p "A serene mountain lake at sunrise, golden light on calm water, photographic style" \
  --cfg-scale 1.0 --steps 8 -H 1024 -W 512 --seed 42 --metadata-format json -v \
  -o outputs/p0-image-run1.png
```

### 5.2 run1 实测结果

| 项 | 实测值 |
| --- | --- |
| 退出码 | 0 |
| 生成阶段耗时 | 182.13 s（`generate_image completed`，image.cpp:1196） |
| 整进程真实耗时 | 190.27 s（/usr/bin/time -l） |
| 模型加载 | 452/453 张量，7.51 s（~480 MB/s，page cache 预热）；参数缓冲 3685.19 MB @ MTL0（GPU） |
| 峰值 RSS | 7,140,900,864 B ≈ 6.65 GiB（/usr/bin/time -l） |
| 采样速度 | 首步 28.1 s/it，稳定约 20 s/it（8 步） |
| 产物 | outputs/p0-image-run1.png，PNG，512×1024，941 KB |
| 内容核对 | 已人工查看图像：晨光山湖、金色水面反光，与提示词语义一致 |
| 进程回收 | 运行结束后 `pgrep sd-cli` 为 0（按可执行路径精确匹配） |

### 5.3 重复运行（run2，同参数同 seed）

- 退出码 0，生成 189.73 s，产物 `p0-image-run2.png`
- **run1 与 run2 的 sha256 完全一致**（`d4ecf12c5d1df7778c1b5380efd0642caa9cfcced9123eb68803c1f79cefad9c`）：同机同参数同 seed 输出逐字节可复现。注意：此结论仅限同一机器/同一引擎 commit，不外推到跨硬件（符合计划"不保证不同硬件上逐像素一致"）。
- 两次连续运行之间无资源干扰，第二次峰值 RSS 与首次同量级。

### 5.4 取消测试（run3，运行中 SIGTERM）

- 启动后 75 s（已过模型加载、处于采样阶段）发送 SIGTERM
- 退出码 143（128+SIGTERM），**终止延迟 0 s**（发送后 1 s 内确认退出）
- 取消后按可执行路径精确匹配检查：残留进程 0
- **取消的运行未写出任何产物文件**（`p0-image-run3-cancel.png` 不存在）——sd-cli 仅在完成时落盘，与计划"中间产物不能登记为作品"的约束天然吻合；生产 runner 仍需自行保证不把失败/取消产物登记为作品

### 5.5 P0-5 小结

Z-Image Turbo（Q4_K 主模型 + Qwen3-4B Q4_K_M 编码器 + FLUX ae VAE）在 M2/Metal 上完成"真实出图 → 重复运行 → 中途取消 → 进程回收"全链路，无遗留进程、无假产物。单张 512×1024/8 步约 3 分钟，作为本机基线记录；未测更高分辨率/步数（留待产品阶段按 profile 展开）。

## 6. 视频最小探针（P0-6）

**状态：已完成。** 经三次尝试（两次内存悬崖中断 + 一次分块解码对照完成）取得可解码产物。以下数据全部为本机实测。

### 6.1 命令（run4，成功版本）

```sh
cd ~/myllama-media-p0
engine/sd.cpp/build/bin/sd-cli -M vid_gen \
  --diffusion-model models/wan2.1_t2v_1.3B_fp16.safetensors \
  --vae models/wan_2.1_vae.safetensors \
  --t5xxl models/umt5-xxl-encoder-Q4_K_M.gguf \
  -p "a lovely cat walking on green grass" \
  --cfg-scale 6.0 --sampling-method euler --steps 10 \
  -W 832 -H 480 --video-frames 17 --flow-shift 3.0 --seed 42 \
  --vae-tiling -v -o outputs/p0-video-probe-tiled.webm
```

（run3 为同命令去掉 `--vae-tiling` 的对照；模型自 run3 起从外置磁盘加载，`--steps 10` 显式指定）

### 6.2 产物验证（ffprobe 实测）

| 项 | 值 |
| --- | --- |
| 容器/编码 | Matroska/WebM，VP8（`--help` 声明的单文件视频输出仅 `.avi`/`.webm`/动画 `.webp`，**无 .mp4**，印证计划 §5.6） |
| 分辨率 / 帧率 | 832×480 / 16 FPS（引擎默认），17 帧 ≈ 1.06 s |
| 可解码性 | `ffmpeg -f null` 全流解码通过，无错误 |
| 内容核对 | 抽取第 9 帧目视：绿草地上的猫，与提示词一致；10 步采样 + 分块 VAE 导致画质偏淡，仅作链路证据不作质量结论 |

### 6.3 关键实测：VAE 解码内存悬崖与分块对照（A/B）

| 指标 | run3（不 tiled） | run4（`--vae-tiling`） |
| --- | --- | --- |
| 采样阶段（10 步） | 684.36 s，RSS ~7.0 GB，空闲内存稳定 34–48% | 同左（684.36 s） |
| VAE 计算缓冲 | 未完成——采样结束进入解码瞬间空闲内存从 38%→9%（~7 GB 抢占，<60 s），两次运行同点位复现（另一次 48%→7%），触发自动终止阈值 | 2180.12 MB（VRAM），解码全程空闲内存 47–75%，平稳 |
| VAE 解码耗时 | 未测得（被终止） | **3573.06 s ≈ 59.5 min**（10 个分块单元，~362 s/单元） |
| 全程耗时 | — | 4282.76 s ≈ 71.4 min |
| 峰值 RSS | 采样期 ~7.0 GB | 7.8 GiB（/usr/bin/time -l） |
| 产物 | 无 | `p0-video-probe-tiled.webm`，253 KB |

结论：M2/Metal 上 Wan 3D VAE 解码是视频链路的**双瓶颈**——不 tiled 则内存爆发（+~7 GB），tiled 则内存有界但耗时爆炸（Metal 对小分块的 kernel 调度开销主导）。上游 build.md 提示的 Metal 大矩阵问题与实际表现一致。视频阶段的真实成本画像：17 帧/10 步/480p 已需 71 分钟，20 步 33 帧（上游文档示例配置）按比例外推数小时——P3 视频正式化必须先解决 VAE 解码路径（更大 tile 的权衡实测、或其他后端）。

### 6.4 一次执行偏差记录

run3 首次启动时漏写 `--steps 10`（默认 20 步），在步 1/20 时发现并主动终止重跑；监控脚本 RSS 采样首版取错进程行（读到 /usr/bin/time 包装进程，显示 0MB），第二次起修正为按进程聚合。两次修正均未影响已采数据的正确性。

### 6.5 运行纪律记录（本节为方法论证据）

- 全程单任务串行，无下载/构建并行叠加
- 分钟级监控（`memory_pressure` 空闲百分比 + 进程 RSS 聚合 + 步数解析）落盘 `logs/vid-probe{3,4}-monitor.log`
- 自动终止阈值：空闲内存 <10% → SIGTERM（run3 两次、以及更早的 run2 共三次触发，全部为 VAE 阶段，机器未出现不可用状态）
- 模型自 run3 起迁移至外置磁盘（KINGSTON，实测读 419 MB/s），仓库侧 `~/myllama-media-p0/models` 为软链接；内置磁盘可用空间由 15 GB 回升至 28 GB
- 每次任务启动前向用户通报预期时长与占用，运行期间用户可随时 `pkill -f "bin/sd-cli"` 自行终止

## 7. 结论与下一阶段准入

- **已验证（图片）**：Z-Image Turbo 组件组合在 macOS/Metal 完成"出图→重复→取消→回收"闭环，确定性（同 seed 逐字节一致）成立，进程零残留，取消无假产物。单张 512×1024/8 步 182 s，峰值 RSS 6.65 GiB。
- **已验证（视频探针）**：Wan2.1 T2V 1.3B 全链路走通并取得可解码 .webm 产物（VP8/832×480/16FPS）；输出格式契约确认（无 .mp4）。**VAE 解码为双瓶颈**：不 tiled 内存爆发 +~7 GB，tiled 内存有界（2.2 GB）但 17 帧需 59.5 min。P3 视频正式化前必须解决 VAE 解码路径；Android 端视频生成本机画像已不支持乐观假设。
- **已验证（组件/工程面）**：6 个组件全部按 HF revision + sha256 锚定下载成功；引擎固定 commit 构建（Metal）可复现；下载链路对限速、SSL 瞬断的容错（aria2 断点续传）有实测记录；模型权重迁移外置磁盘 + 软链接方案验证可用（内置盘释放 13 GB）。
- **Android 准入输入**（详细展开见安卓可行性研究文档）：引擎必须放内部存储（SELinux 禁止外部挂载 exec，paths.go 既有约束），权重可放外部存储——与聊天模型分发同构；图片峰值 RSS 6.65 GiB 需压缩后才可能进入 8–12 GB 设备；视频在本机 GPU 上已需小时级耗时，手机 CPU/OpenCL 无 v1 可行性。
- **平台边界**：本报告仅覆盖 macOS/Metal 单平台单设备；Windows CUDA / Linux Vulkan 未验证，不能由本报告外推。
