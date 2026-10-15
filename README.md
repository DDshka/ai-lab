# ai-lab

A self-contained local LLM lab, run entirely with Docker Compose:

| Service | Image | Role |
|---|---|---|
| `llama-cpp` | `ghcr.io/ggml-org/llama.cpp` | OpenAI-compatible inference server, driven by `llama-cpp-preset.ini` |
| `open-webui` | `ghcr.io/open-webui/open-webui` | Chat UI |
| `open-terminal` | `ghcr.io/open-webui/open-terminal` | Sandboxed shell/code-execution backend for Open WebUI |
| `prometheus` | `prom/prometheus` | Scrapes llama.cpp `/metrics` |
| `grafana` | `grafana/grafana` | Pre-provisioned llama.cpp dashboard |
| `caddy-gateway` | `caddy:2.11.4-alpine` | The only container with published ports |

Two networks: `backend` is `internal: true` (no outbound/inbound access); only `caddy-gateway` straddles `frontend` and publishes ports. Nothing else is reachable from the host except through Caddy.

**Tested under Ubuntu 26.04 running on WSL2 (Windows 11), Docker Desktop engine, NVIDIA RTX 40-series.** Other platforms are untested — the compose file carries notes for Apple Silicon (short version: run llama.cpp on bare metal, Colima can't pass the GPU through).

---

## 1. Minimum setup

### Prerequisites

- Windows 11 + WSL2 with Ubuntu 26.04
- Docker Desktop with WSL integration enabled (`gpus: all` needs the Docker Desktop engine + the NVIDIA driver installed **on Windows**, not inside WSL)
- Disk: model size + ~5 GB for images/volumes

### Files you must create (all are git-ignored)

`.gitignore` excludes `*.env`, `models/` and `user-space/` — a fresh clone has **none** of them. Create all four:

**1. `.env`** — copy the template and fill in the secrets:

```bash
cp .env-template .env
```

| Variable | Notes |
|---|---|
| `LLAMA_CPP_INTERNAL_PORT` | `20000`. Must match the port in `prometheus.yml` (hardcoded there). |
| `OPEN_WEBUI_INTERNAL_PORT` / `_EXTERNAL_PORT` | `10000` / `3000` → UI at `http://localhost:3000` |
| `OPEN_WEBUI_SECRET_KEY` | **Secret.** Session signing key — changing it logs everyone out. |
| `OPEN_TERMINAL_INTERNAL_PORT` / `OPEN_TERMINAL_API_KEY` | `10001` / **secret**. Paste the key into Open WebUI when registering the terminal tool. |
| `GRAFANA_INTERNAL_PORT` / `_EXTERNAL_PORT` | `3000` / `2999` → Grafana at `http://localhost:2999` |
| `GRAFANA_SECURITY_ADMIN_USER` / `_PASSWORD` | **Secret.** Grafana admin login. |

```bash
# generate the two app secrets
printf '%s' "$(openssl rand -hex 32)" # → OPEN_WEBUI_SECRET_KEY
printf '%s' "$(openssl rand -hex 32)" # → OPEN_TERMINAL_API_KEY
```

**2. `llama-cpp-api-key.env`** — a Docker *secret* (not an env file despite the name), mounted at `/run/secrets/llama-cpp-api-key` in both `llama-cpp` and `prometheus`. It is the bearer token for the inference API. Write it with no trailing newline:

```bash
openssl rand -hex 32 | tr -d '\n' > llama-cpp-api-key.env
```

**3. `models/LLM/<your-model>.gguf`** — the directory is mounted read-only at `/app/models`. Add a matching section to `llama-cpp-preset.ini`; the section name is the model id clients and Prometheus use:

```ini
[my-model]
model = /app/models/<your-model>.gguf
load-on-startup = true
n-gpu-layers = 99          ; layers to offload to the GPU; lower it if VRAM is tight
ctx-size = 16000
cache-type-k = q8_0
cache-type-v = q8_0
```

**4. `user-space/`** — git-ignored but **required**; Prometheus and Grafana bind-mount config from it. Recreate:

```bash
mkdir -p user-space/{open-terminal,prometheus,grafana/provisioning/{datasources,dashboards},grafana/dashboards}
```

- `user-space/prometheus/prometheus.yml` — see §4
- `user-space/grafana/provisioning/datasources/prometheus.yml` — datasource `http://prometheus:9090`, `uid: prometheus`, `isDefault: true`
- `user-space/grafana/provisioning/dashboards/dashboards.yml` — file provider, `path: /var/lib/grafana/dashboards`, folder `llama.cpp`
- `user-space/grafana/dashboards/llamacpp-grafana-dashboard.json` — the community llama.cpp dashboard

### Pick the llama.cpp image

`docker-compose.yml` defaults to `ghcr.io/ggml-org/llama.cpp:full`, which is a **CPU** build — `gpus: all` and `n-gpu-layers` have no effect under it. For WSL + NVIDIA switch the `image:` line to one of the CUDA tags listed in the comments above it: `:full-cuda` keeps the extra tools (`llama-fit-params`), `:server-cuda13` ships the server only.

### Run

```bash
docker compose up -d
docker compose logs -f llama-cpp     # wait for the model to finish loading
```

Open WebUI → `http://localhost:3000` · Grafana → `http://localhost:2999`

---

## 2. Memory per context size

Total memory = **weights** (§3) + **KV cache** + a fixed overhead of roughly 1 GiB for compute and graph buffers.

Only the KV cache grows with context. Its size per token is fixed by the model's architecture:

```
bytes/token = attention_layers × kv_heads × (key_length + value_length) × bytes_per_element
```

All four values come from the GGUF metadata (`block_count`, `attention.head_count_kv`, `attention.key_length`, `attention.value_length`). Two things commonly shrink the result:

- **Grouped-query attention** — `head_count_kv` is far smaller than `head_count` (often 4–8 vs 24–64).
- **Hybrid / SSM models** — if the metadata has `full_attention_interval = N`, only every Nth layer keeps a growing KV cache; the rest hold a fixed-size recurrent state (typically 0.1–0.3 GiB per slot) that does **not** depend on context length. With 64 blocks and an interval of 4, only 16 layers count.

### KV cache at `f16`

Find your model's per-token figure, then read down its column:

| `ctx-size` | 32 KiB/token | 64 KiB/token | 128 KiB/token | 256 KiB/token |
|---:|---:|---:|---:|---:|
| 4 096 | 0.13 GiB | 0.25 GiB | 0.50 GiB | 1.0 GiB |
| 8 192 | 0.25 GiB | 0.50 GiB | 1.0 GiB | 2.0 GiB |
| 16 384 | 0.50 GiB | 1.0 GiB | 2.0 GiB | 4.0 GiB |
| 32 768 | 1.0 GiB | 2.0 GiB | 4.0 GiB | 8.0 GiB |
| 65 536 | 2.0 GiB | 4.0 GiB | 8.0 GiB | 16 GiB |
| 131 072 | 4.0 GiB | 8.0 GiB | 16 GiB | 32 GiB |
| 262 144 | 8.0 GiB | 16 GiB | 32 GiB | 64 GiB |

*Example:* a 32-layer model with 8 KV heads and 128-wide K/V → 32 × 8 × 256 × 2 B = **128 KiB/token**, so 32K context needs 4 GiB of KV cache. A 64-block hybrid with `full_attention_interval = 4`, 4 KV heads and 256-wide K/V → 16 × 4 × 512 × 2 B = **64 KiB/token**.

### Quantising the cache

`cache-type-k` / `cache-type-v` in the preset scale the whole table:

| `cache-type-*` | Multiplier vs `f16` | Note |
|---|---:|---|
| `f16` | 1.00× | Default |
| `q8_0` | 0.53× | No practical quality loss — the preset's choice |
| `q5_1` | 0.38× | Mild degradation on long contexts |
| `q4_0` | 0.28× | Noticeable; use only when memory-starved |

### Other things that move the number

- **Multiply the KV cache by `parallel`.** Each concurrent slot gets its own full cache (and its own recurrent state on hybrid models). The preset uses `parallel = 1`.
- **`n-gpu-layers` decides the VRAM/RAM split.** Layers you don't offload stay in system RAM; the KV cache follows its layers.
- **`batch-size` / `ubatch-size`** (1024 / 512 in the preset) add a compute buffer proportional to `ubatch-size`; that is part of the ~1 GiB overhead.
- **Don't guess — measure.** llama-server prints the real figures while loading (`KV self size = …`, `compute buffer size = …`):
  ```bash
  docker compose logs llama-cpp | grep -iE 'kv self|buffer size|model size'
  ```
- **Or size it automatically** — see `llama-fit-params.template`, which fits a context to a memory budget in MiB:
  ```bash
  docker compose exec llama-cpp ./llama-fit-params \
    --model /app/models/<your-model>.gguf --fit-target 512 --fit-ctx 16000
  ```

---

## 3. Quantisation and its effect on size

Quantisation compresses the *weights*. It is independent of the KV-cache quantisation above, and it is the single biggest lever on memory: file size on disk ≈ resident weight memory. All figures below are GiB.

| Format | ≈ bits/weight | GiB per 1B&nbsp;params | 8B model | 27B model | 70B model |
|---|---:|---:|---:|---:|---:|
| `F16` | 16.0 | 1.86 | 14.9 | 50.3 | 130 |
| `Q8_0` | 8.5 | 0.99 | 7.9 | 26.7 | 69 |
| `Q6_K` | 6.6 | 0.77 | 6.1 | 20.7 | 54 |
| `Q5_K_M` | 5.6 | 0.65 | 5.2 | 17.6 | 46 |
| `Q4_K_M` | 4.8 | 0.56 | 4.5 | 15.1 | 39 |
| `IQ4_XS` | 4.25 | 0.49 | 4.0 | 13.4 | 35 |
| `Q3_K_M` | 3.9 | 0.45 | 3.6 | 12.3 | 32 |
| `IQ3_XXS` | 3.1 | 0.36 | 2.9 | 9.7 | 25 |
| `Q2_K` | 2.6 | 0.30 | 2.4 | 8.2 | 21 |
| `IQ2_XXS` | 2.1 | 0.24 | 2.0 | 6.6 | 17 |
| `IQ1_S` | 1.8 | 0.21 | 1.7 | 5.7 | 15 |

Choosing one:

- **`Q4_K_M` is the usual sweet spot** — roughly a quarter of `F16` with little practical quality loss.
- `Q6_K` and above are effectively indistinguishable from the original; below `Q3` degradation is obvious.
- **`IQ*` formats** are quantised against an importance matrix: better quality per bit than the equivalent `Q*`, at some CPU-inference speed cost. "Dynamic" repacks (e.g. Unsloth's `UD-` prefix) keep sensitive tensors at higher precision, which is what makes 1–2 bit quants usable at all.
- **A larger model at a lower quant usually beats a smaller model at a higher quant, down to about `Q3`.** Below that the ordering flips.
- Fewer bits also means less memory bandwidth per token, so lower quants generally generate faster.

---

## 4. Prometheus

**Enablement.** `metrics = true` in the `[*]` section of `llama-cpp-preset.ini` makes llama-server expose `/metrics`. Without it the scrape returns 404.

**Config** — `user-space/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 10s

scrape_configs:
  - job_name: llama-cpp

    authorization:
      type: Bearer
      credentials_file: /run/secrets/llama-cpp-api-key

    params:
      model: [ "my-model" ]   # ← must be a [section] name from llama-cpp-preset.ini

    static_configs:
      - targets: [ "llama-cpp:20000" ]
```

Three things to get right:

1. **`params.model`** ships as the placeholder `specify-model-name-from-preset-here`. Because the server runs in multi-model mode (`--models-preset`), `/metrics` needs this parameter to select an instance. Set it to the exact `[section]` name in `llama-cpp-preset.ini`.
2. **Auth.** The scrape authenticates with the same Docker secret the server uses — the compose file mounts `llama-cpp-api-key` into the Prometheus container for exactly this. No key in the config file.
3. **Target port.** `llama-cpp:20000` is hardcoded; keep it in sync with `LLAMA_CPP_INTERNAL_PORT`. Add more `targets` entries for additional instances.

**Operations.**

- Retention is 30 days (`--storage.tsdb.retention.time=30d`), stored in the `prometheus-data` volume.
- Prometheus publishes **no host port** (it lives on the internal `backend` network). Reach it via Grafana, `docker compose exec prometheus wget -qO- localhost:9090/...`, or add a `reverse_proxy prometheus:9090` block to the `Caddyfile`.
- Apply config changes: `docker compose restart prometheus`.
- Verify the target is `UP` from Grafana → Explore, or check the scrape in the Prometheus logs.

**Grafana** auto-provisions the Prometheus datasource and the llama.cpp dashboard (folder `llama.cpp`) from `user-space/grafana/`. Useful series: `llamacpp:kv_cache_usage_ratio`, `llamacpp:requests_processing`, `llamacpp:requests_deferred`, `llamacpp:prompt_tokens_total`, `llamacpp:tokens_predicted_total`, `llamacpp:prompt_tokens_seconds`, `llamacpp:predicted_tokens_seconds`.

---

## 5. Sharing files with `open-terminal`

`open-terminal` keeps its working directory in the **named volume `open-terminal-data`, mounted at `/home/user`**. It has no host bind-mount by default, so files must be pushed in.

**A. Ad-hoc copy (no config change).**

```bash
docker compose cp ./report.csv open-terminal:/home/user/      # host → container
docker compose cp open-terminal:/home/user/out.xlsx ./        # container → host
docker compose exec open-terminal ls -la /home/user           # verify
```

**B. Persistent shared folder (recommended).** The repo already reserves `user-space/open-terminal/` for this. Add a bind-mount alongside the volume in `docker-compose.yml`:

```yaml
  open-terminal:
    volumes:
      - open-terminal-data:/home/user
      - ./user-space/open-terminal:/home/user/shared
```

Then `docker compose up -d open-terminal`. Anything dropped into `user-space/open-terminal/` on the host appears at `/home/user/shared` in the terminal, and vice versa.

- Mounting at `/home/user/shared` rather than `/home/user` keeps the container's home intact — a bind-mount over `/home/user` would shadow the volume and hide its existing contents.
- The container runs as a non-root user; if writes fail, `chmod 777 user-space/open-terminal` or `chown` the directory to the container's UID (`docker compose exec open-terminal id`).
- Add `:ro` if you want the terminal to read but not modify host files.

**C. From Windows.** Keep the repo inside the WSL filesystem (not `/mnt/c`) for speed, and reach the share from Explorer at `\\wsl$\Ubuntu-26.04\<path-to-repo>\user-space\open-terminal`.

**D. Through the chat.** Files uploaded in Open WebUI are handed to the terminal tool by the integration — the quickest path for one-off files, and it needs no restart.

---

## Security notes

- `.env`, `llama-cpp-api-key.env` and any other `*.env` are git-ignored — **keep it that way**; they hold the Grafana admin password, the Open WebUI session key and both API keys. Rotate anything that has ever been committed or shared.
- The `backend` network is `internal`, so llama.cpp, Prometheus and the terminal are not reachable from the host or LAN — only Caddy's published ports are. Don't add `ports:` to the other services.
- Caddy currently serves plain HTTP on `localhost`. Put TLS in front of it before exposing the stack to anything beyond the local machine.
