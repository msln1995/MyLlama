The "API Router" page starts, stops and monitors the llama-server service (token speed, uptime and the service log). Once running it exposes an **OpenAI-compatible** endpoint that any client speaking that protocol can use.

## Start, stop, restart

The page follows a "single primary action" layout: while stopped, the main button is "Start Server"; while running it flips to a red "Stop Server", with a restart icon button beside it.

- **Start Server**: scans the model directories, generates presets from each model's parameters, then launches llama-server. Starting fails with an error when the model directories contain no GGUF files at all — download a model first;
- **Stop Server**: terminates the llama-server process, freeing VRAM and memory;
- **Restart**: apply new model parameters or server configuration by restarting.

The status card at the top shows the running state, an uptime badge and the service address (default `http://127.0.0.1:8080` — click it to copy). "Available Models" below lists the models recognized in the model directories.

## Server parameters

Click the gear icon at the top right to open the parameter panel:

| Parameter | Default | Meaning |
| --- | --- | --- |
| Port | 8080 | Service listening port, range 1024–65535 |
| Max Loaded Models | 1 | How many models stay loaded in VRAM/memory at once. It is also how many loads the pre-start memory check bills together: every resident model keeps its own weights + context cache + runtime overhead, and those figures really do add up (see the next row) |
| RAM Cache (MB) | 8192 | Prompt cache kept in system memory; 0 omits `--cache-ram` so the server applies its own default. **Both serving modes only ever lower this value**: the cache grows turn by turn through a conversation while the pre-start memory check can only see the moment of launch. Measured — each distinct prompt is retained in RAM at roughly its KV size per token (~113 MiB for 1,035 tokens), and at the desktop default of 8192 six distinct ~6,890-token requests each added ~756 MiB of RSS with no plateau, so a dozen ordinary requests consume that whole 8 GiB. The budget therefore becomes half of the memory the load leaves free: Android prices the model about to load, desktop router mode prices **everything that can be resident at once** — the sum of the N heaviest entries in the library, N being Max Loaded Models (at its default of 1 that is exactly the heaviest entry). Measured with b11720 on a 24 GB M2 at `--models-max 2`, requesting four models in a row, RSS summed over all llama-server processes walked 109 → 3,211 (big) → 3,663 (big + small-a) → 1,013 (small-a + small-b, big evicted) → 1,010 MiB: every loaded model is its own worker process, two loads really are in memory at the same time, and the next request evicts the least recently used one — so pricing only the heaviest entry understates the bill by a whole load as soon as that setting is above 1. Each load is priced at the context in its preset entry (a model that has never been tuned now also gets an explicit context written into that entry: its trained context when the memory free right now can pay for it (never above the tuner's 131072 ceiling, so an untuned model cannot hold a bigger window than its tuned plan), otherwise the largest rung of the tuner's ladder that fits. That number used to be omitted, and the server then picked one you do not control — measured, a 4B Q4_K_M took 103,936 tokens, 17.0 GiB of process RSS and swap from 3.86 to 6.30 GiB, where stating 16,384 costs the same model 4.83 GiB; and because the budget is priced against the loads that can be resident together, one entry the machine cannot pay for is enough to bound the whole library's cache to 0). Losing a cache hit costs one re-processed prompt; over-committing costs a killed process or a machine that starts swapping. **On a Mac "memory left free" is measured as reclaimable memory** (unallocated + file cache − swap in use), matching Activity Monitor, so the cache is no longer zeroed out because the OS is busy caching files |

The panel is locked while the server is running (the fields read "… stop it to edit") so configuration cannot change underneath a live service.

## Monitoring metrics

- **Generation speed**: the status card embeds a gradient area chart of the decode speed (how fast the model writes) over the last 60 seconds, refreshed every second; thinking and answering both count as decode. While running, the prompt processing speed (prefill — how fast your input is ingested) is shown alongside;
- **Server Log**: the log panel at the bottom shows llama-server's raw output live — the first place to look when troubleshooting. The app writes two of its own lines in here: `[INFO] prompt cache budget …` reports how far the machine's free memory shrank the RAM cache budget, and `[WARN] model X needs about …` means the loads that can be resident at once cost more than the memory free right now (with Max Loaded Models at 1 that is the single heaviest entry in the library). Desktop lazy-loads models and shares one service between them, so an unaffordable entry never blocks the start — it gets this line instead: the service comes up, but loading that model spills into swap and requests can stall. Above a cap of 1 the line quotes the **total for the models that can be resident together** and names the heaviest of them with its own share, because the overflow is not necessarily one oversized model — lowering Max Loaded Models is the other lever. When a context tier would rescue it, the line names that tier too, priced exactly like the load it refused (your KV cache types included, and with the other resident models already subtracted from the memory), so the number you type is the one the app already approves. Lower its context size in Model Settings, or run one-click tuning.

## Calling the service from code

The service speaks the OpenAI Chat Completions protocol, e.g. with curl:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"model-name\",
    \"messages\": [
      { \"role\": \"user\", \"content\": \"Hello, introduce yourself\" }
    ]
  }"
```

In third-party clients (chat frontends, IDE plugins, ...) set the API base URL to `http://127.0.0.1:8080/v1`; the API key can be any placeholder while no authentication is configured.

## LAN access and the API key

Both live on the "Settings" page and take effect the next time the service starts:

- **Server Access Scope**: `Local (127.0.0.1)` restricts access to this machine; `LAN (0.0.0.0)` lets other devices on the same network connect — use this machine's LAN IP as the service address then;
- **API Key**: when set, every inference request must carry this Bearer token:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{ \"messages\": [ { \"role\": \"user\", \"content\": \"Hello\" } ] }"
```

When exposing the service to the LAN, setting an API key as well is strongly recommended.
