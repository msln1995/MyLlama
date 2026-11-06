# MyLlama 本地画图与视频生成：项目分析及实施计划

日期：2026-11-06（Asia/Shanghai）
分析基线：`main` / `262f758`，应用版本 `0.6.1`
状态：设计与实施计划，尚未实施；阶段复选框代表后续工作，不代表已完成能力。

## 1. 目标和结论

在 MyLlama 中增加本地图片、视频生成，同时提供清楚的模型选择和安装界面：模型能力、用途、版本、量化方案、整包大小、本机适配、组件状态及开始创作入口。用户提供的 locmodel 截图是交互参考，其体积、速度评价、显存要求和宣传文案不是本项目的验收依据。

推荐保留 Go + Vue + Wails 架构，聊天继续使用 llama.cpp，媒体通过独立的 stable-diffusion.cpp 适配层执行。首版采用按任务启动 `sd-cli` 子进程的方式，先验证单图生成，再建设完整模型目录和安装流程，最后加入视频。首版不引入 Python/ComfyUI 服务，也不重构现有聊天系统为通用引擎框架。

本轮已完成源码静态分析和上游文档读取；没有安装引擎、下载权重、运行生成、测量速度或证明跨平台支持。当前项目已有基础能复用，但新增功能不是两个页面的工作量，核心是模型包、独立任务状态、资源互斥和产物管理。

## 2. 当前实现与接入边界

下表路径相对仓库根目录，行号对应分析基线；实施时以符号为准。

| 当前实现及证据 | 可复用部分 | 必须新增或调整的边界 |
| --- | --- | --- |
| `go.mod`、`frontend/package.json`、`main.go`：Go 1.25、Wails v3 beta.16、Vue 3/TS/Vite；前端资源编译嵌入 | 桌面壳、主题、绑定体系 | 独立媒体绑定与页面，无须换技术栈 |
| `core/gguf.go:239` 扫描目录，选择首个非 mmproj 的有效 GGUF | 文件元数据读取的一部分 | 此入口不能区分聊天与扩散 GGUF；媒体目录必须独立，加入防误纳入规则及回归测试 |
| `core/preset.go:330` 的 `generateModelsPreset` 消费 `scanModels()` | 保留聊天流程 | 不把媒体权重写入 llama-server INI，不复用 LLM 参数结构 |
| `core/tasks.go:19` 的 `DlTask`，`:51` 的 `PersistedDlTask` | 续传、重试、取消、进度及持久化模式 | 扩展资产归属与精确版本恢复；生成任务不能套下载状态机 |
| `core/bridge.go:824` 的 `startHFDownload`，`:894` 目标目录选择；`core/config.go:612` 下载恢复 | HF/ModelScope 文件传输基础 | 入口目前绑定 LLM 根目录；媒体包跨仓库、跨组件下载需要专用入口 |
| `core/llamacpp.go`、`core/archive.go` | 发布资产处理经验、安全解压、下载基础函数 | 独立 sd 引擎版本、平台资产与能力检测；不能直接调用 llama 专用资产选择器 |
| `core/server.go:21`、`core/bridge.go:214` 的 `spawnServerProcess` | 子进程日志与退出回收方法 | llama 单例状态不能给媒体复用；媒体自有 runner、退出通知、日志 |
| `core/config.go:102`、`:742`，`core/paths.go:162`，`core/fsutil.go:246` | 配置迁移、平台路径、原子写 | 应用配置只加少量媒体设置；任务/作品单独持久化 |
| `frontend/src/lib/chat.ts:564` 直接 fetch llama-server；`core/controlplane.go:550` 可启动服务 | 保留已有 API | 仅 UI 禁用不能互斥，必须在后端所有服务启动入口建立共同准入 |
| `core/app.go:162` 的 Shutdown、`:431` 的 SetApiRouteMode、`core/handover.go:53` | 原有 llama 退出和交接行为 | headless 交接仅认识 llama；首版媒体仅 GUI，阻止活动媒体任务期间切换 |
| `frontend/src/views/Models.vue:58`、`router/index.ts` | 模型页标签框架 | 当前标签仅 local/其他二选一，增加媒体标签需改为显式路由匹配 |
| `frontend/src/wails.ts:438`、`views/Downloads.vue` | 搜索浏览 UI 可保留 | `GetModelMaxFileSize` 是最大 GGUF 文件大小，不能当媒体整包体积 |
| `components/RuntimeSection.vue`、`TaskDock.vue` | UI 视觉语言、任务摘要入口 | 原组件绑定 llama/下载/router 状态；新增独立媒体引擎卡和生成状态 |
| `lib/navigation.ts`、`components/MobileNav.vue`、`lib/platform.ts` | 共用导航定义、响应式能力 | 手机直接遍历五项导航，不直接塞第六项；窗口尺寸与 OS 能力分开判断 |
| `core/router.go:23`、前端 LoadedModel.type | 既有模型类型显示 | image/video 类型表示 router 模型分类，不等于已有图片/视频生成实现 |

### 2.1 需要优先处理的隐含风险

1. 聊天和外部 API 可以在 llama-server 存活时重新加载模型，单纯 UnloadModel 不能确保释放资源后不再占用。
2. `stopServerInternal` 在退出等待超时路径仍返回 nil（`core/bridge.go:763` 附近）。媒体准入不能把 nil 当成进程已经退出的证明；需观察退出完成和进程身份，未知状态继续阻止新推理。
3. `Shutdown` 在模式切换时提前返回。媒体退出清理必须覆盖该分支；切换门禁应在修改偏好和拉起继任进程之前生效。
4. 下载状态恢复依赖来源、仓库、文件。媒体包还需要固定 revision、组件身份和完整性数据，不能重启后悄悄下载同名新文件。
5. 主模型、编码器、VAE 不一定来自同一个仓库，也不一定使用同一种量化。一个“Q4”标签不能代表整个包的精度。
6. `make check` 的前端部分当前只有 `npm run build`，需额外执行 `npm test`。不在本轮顺带修改质量脚本。

## 3. 上游证据与技术验证范围

2026-11-06 通过上游 raw 文档读取核对：

- stable-diffusion.cpp README 列出图片与视频模型及 Metal/CUDA/Vulkan 后端。
- `docs/z_image.md` 示例分别传入主模型、VAE 和文本编码器。
- `docs/wan.md` 明确视频需要 `-M vid_gen`，仅设置帧数不等于进入视频模式。
- `docs/minimax_h3.md` 要求专用主模型、视频 VAE、音频 VAE、定制文本编码器，不能随便用同系列通用编码器替换。
- `docs/ltx2.md` 有 `.webm` 输出示例；不能从“视频生成”推断所有构建都直接输出兼容的 MP4。
- `docs/build.md` 的 Metal 章节提示大矩阵运算性能问题；文档能力不能代替目标机器性能测试。

上游来源如下；这些是可变 master 文档，P0 必须固化实际采用的 tag/commit、资产校验值和模型 revisions。此次 GitHub API 查询提交和 release 元数据返回 403 rate limit，未确认最新发布包名称或官方预编译资产覆盖，不据此承诺一键安装已可行。

```text
https://github.com/leejet/stable-diffusion.cpp
https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/README.md
https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/docs/z_image.md
https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/docs/wan.md
https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/docs/minimax_h3.md
https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/docs/ltx2.md
https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/docs/build.md
https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/examples/cli/README.md
```

### 3.1 模型与平台分批开放

| 范围 | 计划 | 开放条件 |
| --- | --- | --- |
| 第一套图片模型 | 优先验证 Z-Image Turbo 的一套明确组件组合；不预先写死量化和大小 | 实际引擎成功生成、取消、重复运行且无遗留进程 |
| 第一套视频模型 | 优先验证 Wan2.1 1.3B 短片；随后验证 Wan2.2 5B 图生视频 | 输出可解码、参数约束明确、耗时和内存有记录 |
| 扩展图片 | Qwen-Image、FLUX、图片编辑模型逐个适配 | 每个版本单独确认组件、许可、能力和参数 |
| 有声视频 | LTX、MiniMax-H3 后续批次 | 音视频输出、同步、组件与内存均完成实测 |
| macOS Apple Silicon | 首个验证平台 | 固定引擎构建、真实设备证据 |
| Windows CUDA / Linux Vulkan | 后续独立验证后开放 | 各平台安装、生成、取消、退出、播放验证通过 |
| Intel Mac / CPU | 暂不承诺可用体验 | 独立性能验证，不能继承 Metal 结果 |
| Android | 首版不开放本地媒体执行和引擎安装 | 单独研究打包、进程、内存与后台限制；仍保证现有 Android 构建与功能不回归 |

候选清单不是首版全部支持清单。不直接照搬截图的所有模型和“流畅”标记。设备实际内存、可用磁盘和已存在权重在 P0 执行前重新核查。

## 4. 产品信息架构

### 4.1 页面与入口

- 保留 `/models/download`、`/models/local`，新增 `/models/media`，其中用“画图 / 视频”过滤；媒体安装状态在本页展示，原本地聊天列表保持 LLM 语义。
- 桌面侧栏增加“创作”，拟定 `/create/image`、`/create/video`、`/create/works`。媒体卡的“开始创作”携带模型包和版本 ID。
- 手机尺寸保留五个主导航，从模型媒体页进入创作并提供返回入口；使用导航 selector，避免复制两套独立菜单。小尺寸桌面窗口仍可创作，Android 的禁止执行由后端能力决定。
- 运行环境增加 `MediaRuntimeSection`，与现有 llama 引擎卡并列。模型页、创作页引用同一运行时状态。
- TaskDock 显示生成任务摘要、取消和跳转；完整参数、日志和作品放创作页。切换标签或路由不得取消后台任务。
- 每页遵循现有主题 token、中英文、三档布局；keep-alive 页面需正确停止/恢复轮询。

### 4.2 模型卡字段及来源

| 字段 | 来源与显示规则 |
| --- | --- |
| 名称、开发者、版本、日期 | 人工维护并审核的内置目录；版本日期不明则不显示，不猜测 |
| 能力 | modelCapabilities、engineCapabilities、appCapabilities 取交集得到当前可用操作；模型固有能力可在详情说明 |
| 用途与限制 | 双语审核文案，避免“最准/最强/流畅”等无证据断言 |
| 量化选项 | 有效 bundle variant 列表；下拉主标签可写主模型量化，展开显示其他组件精度 |
| 整包体积 | 所选 variant 必需组件按资产身份去重后的字节合计；未知组件大小显示“待确认” |
| 本次下载量 | 扣除本地已验证可复用文件及可续传字节；分别显示“完整包”和“还需下载” |
| 磁盘预检 | 完整安装占用、临时/解压空间、生成输出预算；不能只检查下载流量 |
| 推荐版本 | 硬件与固定生成 profile 的规则结果，并显示推荐理由和证据等级；允许手动选择受支持版本 |
| 本机适配 | “已实测”“预计可运行”“资源不足”“未验证”；未知显存不可按 0 或足够处理 |
| 安装完整性 | 未安装、下载中、部分就绪、校验失败、可使用；组件齐备但引擎不兼容仍不能开始 |
| 操作 | 安装引擎、下载整包、暂停/续传、补齐组件、导入、开始创作，依据状态决定 |

推荐与耗时必须绑定：设备/后端、引擎版本、模型 variant、宽高、步数、帧数、FPS、offload 策略。Apple 统一内存单列预算，不能把它与“独立显存 + RAM”重复相加。首版不自动在用户机器上启动测速任务。

## 5. 后端设计

### 5.1 模块拆分（拟定名称）

| 新增模块 | 职责 |
| --- | --- |
| `core/media_catalog.go` + `core/media_catalog/*.json` | 内置有 schemaVersion 的审核目录，校验模型/variant/组件引用 |
| `core/media_models.go` | 安装包清单、组件校验、导入匹配、ready 状态 |
| `core/media_runtime.go` | sd 引擎安装、版本、后端、能力探测；安装到版本化目录 |
| `core/media_download.go` | 组件下载计划、共享组件去重、下载队列桥接与失败恢复 |
| `core/media_jobs.go` | 独立任务状态机、排队、持久化和重启恢复 |
| `core/media_runner.go` | 固定版本的命令适配、进程/日志/取消/退出；不同模型族构造器保持边界 |
| `core/media_outputs.go` | 产物校验、作品记录、预览/导出与目录内文件访问 |
| `core/inference_arbiter.go` | llama 与媒体的原子准入、占用归属及退出后释放 |

共用底层下载和解压需要小范围抽取时，保持现有 llama 行为与测试；不建立新的通用 `engine.go`，不继续把媒体业务塞进 `app.go` 或 `bridge.go`。

### 5.2 数据契约

- `MediaModel`：稳定 ID、系列/版本、双语简介、来源、许可、固有能力、variants。
- `MediaVariant`：稳定 ID、主模型量化、组件集合、runner profile、引擎兼容范围、平台验证记录。
- `MediaComponent`：角色、来源仓库、revision、文件、size、hash、格式；支持 diffusion / high-noise diffusion / text encoder / vision encoder / video VAE / audio VAE 等角色，不固定成只有三件套。
- `MediaInstall`：包与 variant、已解析的组件身份/本地路径/校验结果、目录版本。共享组件引用必须防止某个包删除时误删其他包依赖。
- `MediaJob`：ID、请求幂等 ID、模型/variant、引擎和目录快照、参数、输入图片副本、状态、阶段进度、创建/开始/结束时间、错误、日志和产物引用。
- `MediaArtifact`：任务 ID、输出类型、相对路径、大小/hash、尺寸、时长/FPS、音轨信息、缩略图与参数记录。
- 生成参数按能力显式开放：提示词、seed、宽高、steps、采样器，以及适用模型才有的 negative prompt、CFG、输入图片、帧数/FPS。模型族约束在后端校验，不能只依赖控件范围。

配置仅增加独立引擎/模型/作品目录偏好，缺省字段兼容旧配置。任务和作品写入独立版本化文件，使用原子写、节流进度写入和失败可见性。日志不放进主配置。

目录建议：通过现有平台路径解析派生 `media-runtime/`、`Media-Models/`、`Media-Outputs/`；大文件目录可在设置中调整。不能嵌套在聊天模型根目录，禁止把媒体目录再次作为聊天导入目录；对已受管媒体文件补扫描排除。未知来源 GGUF 不能单凭扩展名认定用途。

### 5.3 模型包安装与本地导入

1. 先解析固定版本模型包，展示必需组件和总量，再确认可运行的引擎组合。
2. 每个组件独立续传；同一资产并发请求合并，固定 revision 和校验值进入持久化队列。新字段缺省为旧 LLM 任务，历史任务能继续恢复。
3. 下载到 `.part`，完成校验后原子发布；所有必需组件合格才将包标记 ready。单组件失败保留其他成功组件，只重试缺失部分。
4. 导入先输出匹配报告：识别结果、可复用文件、缺件、预计新增下载。文件名只作候选，不能据文件名自动认定匹配。
5. 首版仅承诺匹配内置目录的导入；未知模型显示“不支持或需选择已知配套”，不承诺任意 GGUF/safetensors 自动补齐。
6. 外部文件默认只引用、不移动、不删除；记录文件身份并在使用前校验。管理的复制文件与外部引用在卸载时分别处理。
7. HF 与镜像/ModelScope 只有在组件身份和校验一致时可切源；许可或鉴权受限时明确提示，不能显示安装成功。
8. sd 安装先暂存、校验、探测再切换 active 版本；存在生成任务时禁止替换引擎。macOS 签名/隔离与 Windows DLL、Linux共享库在平台验收中验证，不绕过系统保护。

### 5.4 生成状态与进程边界

正常路径：`queued → preparing → running → finalizing → succeeded`。失败进入 failed；取消进入 cancelling，在进程退出并回收后进入 cancelled；应用意外退出后的活动任务恢复为 interrupted。首版排队任务重启后也要求用户显式重试，不自动消耗资源。

- 下载恢复只恢复字节传输；生成重试从头开始，不承诺采样断点恢复。
- 一个任务仅启动一次，重复请求返回原任务；取消和完成竞争必须有唯一终态。
- 默认单任务串行，排队长度有界。进度无法可靠解析时显示阶段和已耗时，不伪造百分比或 ETA。
- 以参数数组启动已验证的引擎，调用 `hideWindow`，不使用 shell 拼接；设置工作目录、受控输出路径与完整错误记录。
- 设置可解释的任务超时与有界退出等待；按进程身份清理子进程，不按进程名广泛终止。退出不确认时保留资源占用，不放行下一任务。
- 成功必须同时满足进程成功退出、预期文件存在、格式与尺寸/帧/音轨检查通过。中间文件不能登记为作品。
- 每任务保留输入与参数、引擎版本、组件 hash、命令参数和日志；不保证不同硬件上逐像素一致。

### 5.5 资源互斥与生命周期

首版采用可证明的保守策略：本应用管理的 llama-server 存活时禁止媒体执行；媒体准备/生成/收尾及退出未确认时禁止 llama 启动。单一 arbiter 原子持有 reservation，覆盖正常启动、自动启动、控制平面、进程收养及模式切换，不能“检查完再启动”留下竞态。

UI 提供“先停止聊天/API 服务”的明确路径，说明会中断现有请求，用户操作后才停服。停止返回 nil 还需确认实际退出。首版不自动重启聊天服务，也不把卸载模型当停服。

媒体运行期间 `SetApiRouteMode` 在写偏好前拒绝切换。正常退出取消媒体并有界等待，重启按 interrupted 展示；媒体不写入现有 llama handover 文件。原有 llama GUI ↔ headless 无中断交接保持回归覆盖。

该互斥仅管理 MyLlama 自己的引擎；其他应用的 GPU/内存占用只能检测和提示，不能承诺系统级独占。未来若支持聊天/媒体并发，须新增显存预算与 API 请求协调方案，不能只删除门禁。

### 5.6 视频输出与作品展示

先验证固定引擎的原生输出格式；若其输出无法在目标 WebView 中播放，再引入固定版本编码器或导出转换步骤，并计入安装、许可、打包和取消清理范围。不能默认增加 ffmpeg 就算视频支持完成。

视频验收包含宽高、帧数、FPS、实际时长、可解码性、首尾帧和播放；有声视频另验音轨、时长与同步。原始产物和转换成品分开记录，禁止改扩展名冒充转码。

`main.go` 目前仅服务嵌入前端资源。作品预览需要专门设计受控 asset handler 或本地资源接口，用作品 ID 映射受管文件、支持正确 MIME 和视频 Range，不暴露任意磁盘路径，也不通过绑定返回整段视频 base64。缩略图列表分页/懒加载；大文件播放、定位、导出在各平台 WebView 验收。

## 6. 前端与绑定实施范围

拟新增 `MediaModels.vue`、`CreateImage.vue`、`CreateVideo.vue`、`MediaWorks.vue`，共用 `MediaModelCard`、`MediaComponentList`、`MediaRuntimeSection` 和任务摘要组件。参数表单按能力渲染，不把各模型所有高级参数一次铺开。

新增纯模块 `lib/mediaCatalog.ts`、`mediaJobs.ts`、`mediaCompatibility.ts` 负责派生状态；后端是安装和任务状态的权威来源。首版使用可恢复查询轮询，切页再查完整状态，不依赖仅存在内存的事件。

拟定绑定：GetMediaRuntime、ListMediaModels、PlanMediaInstall、InstallMediaVariant、InspectMediaImport、ImportMediaComponents、StartMediaJob、CancelMediaJob、GetMediaJobs、ListMediaArtifacts、ExportMediaArtifact。实际签名在 P1 固定，长任务返回 ID，不阻塞绑定等待生成结束。

同一批变更同步 `core/app.go` → 自动生成 bindings → `wails.ts` → 页面 → `dev/mockRuntime.ts` 数字 ID → `dev/mockData.ts`。Mock 必须覆盖引擎缺失、缺件、未知硬件、下载失败、取消、退出中、interrupted 和产物失败，不以未知调用静默成功。

用户可见变化同批更新中英文教程、i18n、README相关流程和截图；新教程章节同步 manifest，不改旧章节路径。首版 Android 不开放执行，仍必须验证其平台门禁及现有页面布局。

## 7. 分阶段任务包与验收

所有阶段先审查实际 diff 和聚焦测试，失败先修复，再跑全门禁。以下是实施顺序，不能把 UI mock 完成标成引擎功能完成。

### P0：固定技术基线，真实生成验证

- [ ] 盘点实际硬件、磁盘、可复用模型；确定下载体积和实验输出目录。
- [ ] 固定 sd 引擎版本/构建来源、模型 revisions/hash、许可与参数，探测实际 `--help`。
- [ ] 图片候选跑出真实图片，验证取消、重复运行和进程回收。
- [ ] 做最小视频探针，记录输出格式、时长、资源和耗时，为视频阶段判断风险。
- [ ] 保存可复现命令、日志、产物元数据和结果报告；模型、二进制和大产物不入 Git。

允许范围：实验脚本和 `docs/` 验证记录；不改现有聊天生产流程。验收：图片证据完备后可进入图片开发；视频探针不通过则记录阻塞，视频保持实验状态。实际发布资产未确认时不进入一键安装承诺。

### P1：模型目录、独立 runner 与任务底座

- [ ] 建立目录/模型包/任务/作品数据契约、校验、独立路径和旧配置兼容。
- [ ] 实现固定图片 profile 的命令构造、手动配置引擎和已知模型导入。
- [ ] 建立任务状态、幂等、持久化、取消、退出回收及产物校验。
- [ ] 接入原子资源互斥和模式切换门禁，明确未知退出状态。
- [ ] 增加最小调用入口、生成 bindings/mock，同批完成接口测试。

允许范围：上述新增后端域、必要 app/config/paths/bridge 生命周期接点、wails/mock、相关测试。禁止：更改聊天协议、对外新增媒体 API、扩大 headless 媒体能力、重写调参器。

验收：已知模型从本地导入到单图文件闭环；并发 start/generate 只有一方成功；取消/崩溃/超时不会产生假成功或错误释放资源；历史下载恢复和聊天启动测试通过。

### P2：模型目录 UI、一键安装与图片创作正式闭环

- [ ] 固定版本的 sd 安装器、组件包下载/校验/补件/恢复/去重。
- [ ] 媒体模型卡、variant 切换、实际下载量、解释性适配标签、本地导入报告。
- [ ] 画图页、作品预览/保存/参数重用、TaskDock 摘要。
- [ ] 路由/导航/引擎卡/设置、中英文教程、mock 与截图同步。
- [ ] 干净配置从安装引擎、下载完整包到生成和查看图片全流程验证。

允许范围：P1相关模块、下载入口/恢复、媒体前端组件、绑定、教程和测试。禁止：复制扩散权重到 LLM preset、把图片生成塞入下载状态机、未经验证开启所有候选模型。

验收：断网、磁盘不足、损坏文件、缺件、变体切换、重启续传可解释且可恢复；正常用户无需手工拼组件即可出图。至少一个平台/模型组合有真实证据，其余明确未验证。

### P3：视频生成闭环

- [ ] 基于已验证 Wan profile 增加视频参数及输入图片能力，落实帧数/尺寸/时长约束。
- [ ] 验证长任务阶段进度、取消、磁盘预算、收尾和播放/导出。
- [ ] 必要时加入独立编码步骤与依赖管理，覆盖转换失败/取消/音轨保留。
- [ ] 作品页支持视频播放、拖动定位、缩略图及参数追溯。
- [ ] 视频教程和各关键状态截图同步。

允许范围：媒体 runner/profile/outputs/jobs、视频 UI、相关依赖打包与测试。禁止：首批同时扩展所有有声大模型、承诺生成断点恢复。

验收：真实文生视频文件通过解码、帧数/FPS/时长验证；图生视频只有在独立验证后开放；Wails 内实际可播放、取消无孤儿进程、失败无假作品。

### P4：模型及平台扩展

- [ ] Qwen/FLUX、编辑、多专家模型逐个增加适配和对应组件测试。
- [ ] LTX/MiniMax-H3 有声视频独立验证音轨与专用编码器。
- [ ] Windows CUDA、Linux Vulkan 各自完成安装到导出验证；补齐发行依赖和许可证。
- [ ] 基于可追溯实测丰富推荐，保留“未知/未验证”状态。

验收：每新增组合均有平台、引擎、组件版本、参数和产物证据。Android 需另立方案，不算桌面扩展自然完成。

## 8. 验证矩阵与交付证据

| 层级 | 必测内容 |
| --- | --- |
| 目录/参数纯函数 | variant 引用、组件去重、unknown size、整包/剩余量、能力交集、参数合法范围、未知硬件 |
| 下载/持久化 | 旧配置、固定 revision 恢复、Range 不受支持、损坏文件、跨源身份、部分包、共享组件、导入外部文件不被删除 |
| runner/状态 | 用可控测试子进程验证成功/非零退出/挂起/取消竞态/假成功文件/路径越界/重复请求/异常重启 |
| 资源/生命周期 | GUI/控制平面并发启动、进程收养、外部 API、停止超时、取消未退出、headless 拒绝且不改偏好、应用退出 |
| 前端 | 模型卡与创作页状态一致、轮询生命周期、变体切换、禁用理由、错误重试、mock 未知 ID 拒绝 |
| 真实 UI | 320/430/768/1100+ 宽度，明暗主题、中英文；空/加载/错误/缺件/成功状态；TaskDock 不遮按钮；实际截图复核 |
| 真实硬件 | 完整模型包、固定引擎，图片/视频输出验证，耗时/峰值资源/取消/重复生成；不同 OS 的 WebView 播放 |
| 原有功能回归 | 聊天自动启动、LLM 下载恢复、router preset、模型导入、API、GUI/headless 交接、Android 构建与页面 |

实施阶段跨绑定、状态或配置变更属于全门禁：

```sh
# 仓库根目录；先满足 go:embed 依赖并重新生成绑定
wails3 task build:frontend
make check
(cd frontend && npm test)
# 新增并发准入和状态机后，运行相关 backend race tests
# 具体 -run 名称随新增测试确定；不得用空匹配冒充通过。
git diff --check
```

Windows 使用 `scripts/check.ps1` 并核对其中是否包含前端测试；未包含时同样补跑 `npm test`。真实 UI/引擎/硬件验收单独记录，不能由单元测试或 mock 替代。全套测试通过后不重复无关测试。

每阶段交付：实际修改范围、设计决定、命令与结果、失败/限制、版本和设备矩阵、证据路径、下一阶段准入结论。生产实现、测试和教程在同一任务范围内闭环；大模型/二进制/本机配置/日志不提交，新的运行目录及时加入 `.gitignore`。

## 9. 工作组织与后续决策

建议每次只推进一个可运行阶段。模型目录审核与 UI mock 可并行，runner 与生命周期由同一责任人整合；同一 GPU 的图片/视频实验串行执行。每个实施包明确文件所有权，主线程审查真实 diff 并亲自执行决定性验证。

实施前必须由 P0 解决：准确引擎版本和资产、首套组件与大小、实际目标设备预算、输出容器/编码策略。后续产品决策包括是否开放媒体 HTTP API、并发聊天、自动测速和 Android 本地执行；这些均不阻塞首版桌面单图闭环。

暂不设日历工期：预编译资产、模型组合、Metal 性能与视频编码链路尚未实测。P0 之后按任务包重新估算，避免用页面开发时间代替整套功能成本。

## 10. 本轮交付记录

- 已完成：两路只读源码分析、主线程关键路径复核、官方原始文档读取、本文落盘。
- 未执行：生产代码变更、引擎/模型安装、真实推理、性能测试、应用截图、远程 issue/push。
- 本文为文档变更，按仓库文档门禁检查 whitespace、内部引用和内容完整性；不运行与纯计划无关的编译测试。
