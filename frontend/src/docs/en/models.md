The "My Models" tab of the Models page gathers every usable model on this machine, and each model can have its own inference parameters. When the API service starts, the app turns these parameters into a llama-server preset per model.

## The model list

The list merges two sources, distinguished by a label:

- **Download Path**: models downloaded through the Download tab;
- **External Path**: directories imported via "Choose Folder".

Each model card shows the name, architecture (e.g. `qwen2`), quantization (e.g. `Q4_K_M`), size and full path. Only real GGUF models make the list: a file that carries a `.gguf` name but is not a GGUF file (a Git LFS pointer, the HTML error page a failed download left behind) is skipped — llama-server could never open it, and letting it in would take the whole service down with it. Multimodal-capable models carry a "Multimodal" badge (an mmproj file was found next to them). When the directory holds several projectors, the one that loads — and the one counted against memory — is the first by name; the rest cost nothing. A projector's size is included both in the settings page's memory estimate and in the pre-start memory check. Click "Refresh" to rescan after files change.

## Opening model settings

Click the gear icon on a model card to open that model's dedicated settings page. Parameters are organized into six tabs — Basic params / Inference / Memory & loading / Multi-GPU / Long Context / Advanced — each with its own usage hint.

## One-click auto-tune

The "Auto-tune" button sits at the top right of the settings page; you do not need to understand any parameter to use it:

1. Click it once — the app reads the model GGUF's real metrics (layer count, attention head structure, KV cache geometry, trained context, MoE expert sizes) plus a hardware snapshot (GPU VRAM, RAM, CPU cores, measured memory bandwidth) and computes an optimal parameter set;
2. The result **fills the form in real time** — every field (GPU layers, context size, cache type, CPU threads, ...) updates immediately so you can review and hand-tweak anything before saving;
3. Click "Save" when satisfied.

Tuning plans VRAM against the **inference GPU** chosen in Settings: on multi-GPU machines, whichever card you picked is the one whose VRAM is budgeted. Possible plans include full GPU offload (fastest), cpu-moe for MoE models (experts stay in RAM, the rest goes to the GPU), partial offload (fallback when VRAM runs out) and CPU-only (no GPU), with the largest context that fits the budget. On Apple Silicon (macOS arm64) the Metal plan keeps every layer GPU-resident via unified memory, sizes the context from the RAM budget while never passing the Metal device's own working-set ceiling (a 24 GB M2 reports 18186 MiB; past it the load does not fail loudly — the server reports the model loaded and passes /health, then answers every request with Compute error until the model is reloaded), and charges the runtime reserve at the Metal measurement (320 MiB at μBatch 512, a quarter of the 1280 MiB CPU figure: with every layer and the whole cache on the GPU no CPU graph is left to hold the batch buffers), so the same model affords a larger context tier than the CPU figure allows — a Mac load that pins GPU layers to 0 is still priced the CPU way. Flash Attention is left to llama-server's automatic choice (measured no slower than forcing it on); on hybrid chips (Apple Silicon, Android SoCs) the thread count is capped at the performance-core count; and if both KV cache types are already `q8_0`, the plan keeps that pair and sizes the context from the measured 0.53× cache footprint (the saved memory becomes a larger window instead of only headroom); on Linux, AMD / Intel GPUs are GPU-accelerated through the Vulkan build too.

Auto-tune rewrites only the fields it actually computes — GPU Layers, Context Size, CPU Threads, Flash Attention, the KV cache types, cpu-moe and µBatch Size. Everything else you set by hand survives: Batch Size, Load Mode, Split Mode and the per-GPU ratio, Main GPU, RoPE Scaling, the mmproj override, Reasoning Output and Speculative Decoding — pressing Auto-tune again no longer silently wipes a speculation setting. µBatch is priced from your existing value: activation memory grows linearly with it (measured +1.6 GB from 512 to 2048 on the same M2 model), so raising it for faster prefill buys a smaller context instead of pretending the memory is free.

Context checkpoints are chosen the same way and need no setting from you. While it processes a prompt a sliding-window model can keep copies of its window-sized KV resident in RAM, and the app caps that budget at 8 checkpoints instead of llama-server's default 32 — measured on an M2 with gemma-3-1b Q4_K_M at ctx 16384: 32 checkpoints peaked at 1.21 GiB of RSS, 8 at 1.09 GiB, while revisiting a branched prompt stayed just as fast (2.24 s vs 2.26 s). The retention is priced into the context tier the plan picks, into the settings-page estimate and into the pre-start memory check, so no page calls affordable what the guard would refuse. Android plans and Windows full-offload plans run with the budget at 0: for those loads it buys nothing (the chat page only appends turns, which measured identical in speed with checkpoints off and 188 MiB of RAM cheaper) while competing for memory the load itself needs. Plain-attention models have no window to copy, so they are charged nothing.

## Quick parameter reference

| Parameter | Meaning and advice |
| --- | --- |
| GPU Layers | `auto` puts everything into VRAM (recommended); a number does partial offload when VRAM is tight; `0` for CPU-only inference |
| Context Size | how much text the model "sees" at once; larger for longer conversations, limited by VRAM |
| CPU Threads | `-1` for auto; lower it on weak CPUs or when heat/usage is high. On hybrid chips (Apple Silicon, Android SoCs) auto-tune only uses the performance cores — measured M2 4P+4E CPU-only: 21.5 t/s at 4 threads vs 10.0 at 8. On Android that cap now applies even without tuning — `-1` resolves to the performance-core count (see the note below) |
| Flash Attention | Checking it forces `-fa on`, leaving it unchecked lets llama-server choose — measured M2 Metal decode: auto 30.0 ≈ forced on 29.7, off 27.2 t/s |
| cpu-moe | For MoE models (DeepSeek, Qwen3-MoE, ...) low on VRAM: keep expert layers in RAM |
| KV Cache Type | `q8_0` measured at 0.53× the f16 cache size (nearly half), at a slight decode cost — measured M2 Metal: 28.4 vs 30.1 t/s on f16. A mismatched K/V pair is priced as the mean of its halves (`q8_0` / f16 measured 0.77×). The memory estimate and the pre-start memory check price every type from one table, each figure a real memory step measured on an M2: f32 2×, q5_1 0.38×, q5_0 0.34×, q4_1 0.31×, q4_0 / iq4_nl 0.28× (only size was measured for these — speed and quality were not). Setting **both** K and V to `q8_0` is a complete quantized pair: auto-tune then keeps it and prices the context ladder at the 0.53× cache footprint, so the saved memory turns into a larger window (measured 4B Q4_K_M + 24 GB machine priced in the Metal shape: the 64K tier needs about 11.6 GiB free, only about 7.4 GiB with q8_0); default is f16 |
| Load Mode | Default mmap loads fastest; mlock prevents swapping when RAM is ample |
| Split Mode | `none` for a single GPU; `layer` is the stable default for multi-GPU |
| Speculative Decoding | "Advanced" tab. `ngram-simple` / `ngram-mod` are n-gram self-speculation: no draft model, zero extra VRAM; `draft-mtp` is MTP speculative prediction and needs model support; empty disables. Measured M2 Metal 4B Q4_K_M: on highly repetitive output `ngram-mod` is ~2.8x (86-88 vs 31 t/s baseline), on natural prose it gains nothing, and `ngram-simple` costs ~5% either way — enable it for code / templated output only |

> Note: Android is a CPU-only build, so GPU-only parameters (GPU layers, Flash Attention, cpu-moe) are not shown there; the `dio` load mode is only offered on Windows / Linux. Android detects the SoC model automatically (e.g. Snapdragon / Dimensity, read from system properties); on Android and Apple Silicon, auto-tune caps the thread count at the performance-core count (big.LITTLE efficiency cores and Apple Silicon E-cores do not join inference threads), and the tune-result toast for the CPU-only plan no longer shows a GPU-layers field. On Android an unset thread count is capped to the performance cores even without tuning: measured on a vivo X60 (Exynos 1080, 4×A55 + 4×A78) with Qwen3-4B Q5_K_M, 4 threads decode at 3.5–3.9 t/s and finish the same request in 97–103 s, while llama.cpp's own all-8-core default manages 0.35 t/s and 443 s — once the efficiency cores are busy with inference workers, the UI and memory reclaim starve together. The value actually sent is recorded in the server log on the API page (`direct mode: threads -1 -> 4`); devices whose probe finds a single performance core are left on llama.cpp's own choice. The Reasoning Output toggle in the "Advanced" tab takes effect on Android as well and sends `--reasoning off`: thinking tokens decode exactly like the answer, so switching them off is time the phone does not have to spend.

## Changing parameters while the service is running

Presets are generated **when the service starts**. If the service is running, saving shows a note that "saved parameters take effect after the service is restarted" — restart the service on the "API Router" page to apply them.
