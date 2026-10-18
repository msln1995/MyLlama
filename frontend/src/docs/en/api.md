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
| Max Loaded Models | 1 | How many models stay loaded in VRAM/memory at once |
| RAM Cache (MB) | 8192 | Prompt cache kept in system memory; 0 omits `--cache-ram` so the server applies its own default. **Both serving modes only ever lower this value**: the cache grows turn by turn through a conversation while the pre-start memory check can only see the moment of launch. Measured — each distinct prompt is retained in RAM at roughly its KV size per token (~113 MiB for 1,035 tokens), and at the desktop default of 8192 six distinct ~6,890-token requests each added ~756 MiB of RSS with no plateau, so a dozen ordinary requests consume that whole 8 GiB. The budget therefore becomes half of the memory the load leaves free: Android prices the model about to load, desktop router mode prices the heaviest entry in the library (an untuned model at its trained context, since its load carries no `--ctx-size`). Losing a cache hit costs one re-processed prompt; over-committing costs a killed process or a machine that starts swapping |

The panel is locked while the server is running (the fields read "… stop it to edit") so configuration cannot change underneath a live service.

## Monitoring metrics

- **Generation speed**: the status card embeds a gradient area chart of the decode speed (how fast the model writes) over the last 60 seconds, refreshed every second; thinking and answering both count as decode. While running, the prompt processing speed (prefill — how fast your input is ingested) is shown alongside;
- **Server Log**: the log panel at the bottom shows llama-server's raw output live — the first place to look when troubleshooting.

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
