A quick troubleshooting handbook. Try the simplest fix first.

## "no models found in the LLM-Models directory" when starting the service

Before starting, the app scans the model directories (the model download path plus the external path imported on the "My Models" tab of the Models page) and generates presets; the error appears when neither contains any GGUF file. Any one of these fixes it:

- Download at least one model on the "Download" tab of the Models page;
- Or point the "Model Download Path" row on the Settings page at your existing model folder;
- Or import an existing model directory with "Choose Folder" on the "My Models" tab of the Models page (imported models are recognized by the service too).

## "only ... of memory is available" when starting the service on a phone

Android has no swap: once a load needs more memory than the system currently reports as available, it does not fail with an error — the low-memory killer takes the process down, which looks like the app simply crashing without explanation. Before launching llama-server the app works out what that load needs (model weights + the KV cache for the configured context size + the runtime overhead) and, when it cannot fit, shows both numbers as a banner on the API page (also written to the server log) instead of letting the system kill it; a server that is already running is left untouched. Fix it by lowering the "Context Size", picking a smaller quantization (a 4B at Q4_K_M is roughly 2.5 GB), or using a phone-sized model. A model you never configured starts with the app's default 4096-token context rather than the context the model was trained with (hundreds of thousands of tokens of cache — impossible on a phone). The check only applies to Android direct-load mode; desktop still lazy-loads models on demand.

## Generation stops when I leave the app or lock the screen on a phone

Android does not merely deprioritise a backgrounded app — it moves it onto the little cores and can freeze the process outright (measured on a vivo X60: after backgrounding, `Cpus_allowed_list` is left with 0-3, the four A55 efficiency cores, the cgroup becomes `freezer:/frozen` and the task state D). llama-server is a child of the app process and inherits both, so an answer that is still generating stops the moment you switch apps or the screen goes off. While the service runs, the app now starts a foreground service (an ongoing notification naming the resident model), so the process is no longer treated as an ordinary background app and does not get frozen; stopping the service takes the notification away with it. On Android 13 and above the system asks for notification permission on the first start.

## Port already in use

- **Service port (default 8080)**: if starting fails and the log reports the port is taken, check what occupies 8080; for a lasting conflict, change Port to another value between 1024–65535 in "API Router → Settings" (while the service is stopped).
- **Dev port 5173**: under `wails3 task dev` Vite binds 127.0.0.1:5173; when starting dev again reports the port is occupied, end the stale process holding it first.

## The right way to stop the service

Always use the **Stop Server** button on the "API Router" page: it terminates only the llama-server process the app launched. Do not force-kill by image name from outside (e.g. `taskkill /IM llama-server.exe`) — with multiple instances running you would kill other instances too.

## What does the CUDA compatibility notice mean

The "CUDA Compat" row on the System Environment page's CUDA card (System Info tab; shown only on Windows when an NVIDIA GPU is detected) has three states:

- **Compatible**: a regular GPU — the current CUDA runtime works as-is;
- **Requires CUDA ≥12.8**: your GPU is Blackwell (RTX 50 series) but the app cannot prove the current CUDA runtime meets the 12.8 floor. llama-server may then fail with `no kernel image`. **Fix**: re-download the latest llama.cpp in the "Runtime" tab (its bundled CUDA runtime satisfies the floor);
- **Satisfied**: a Blackwell GPU with a conforming runtime — nothing to do.

## What to do when a download fails

- Transient network problems (timeouts, 429, 5xx) are **retried automatically** up to 3 times — usually nothing to do;
- If automatic retries are exhausted, click **Retry** in the download manager or on the task card;
- Downloads are **resumable** — retrying never restarts from zero;
- Persistent failures are usually network-related: on Chinese networks switch to the HF Mirror or ModelScope source and try again.

## Why did my parameter change not apply

Presets are generated **when the service starts**. Parameters saved while the service is running only apply at the next start — when the settings page shows the note, click **Restart** on the "API Router" page.

## Where do I see the service log

The Server Log panel on the "API Router" page shows llama-server's raw output live: startup failures, load timings and API errors all show up there first.
