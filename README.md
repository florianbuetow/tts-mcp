# mistral-text-to-spech

![Made with AI](https://img.shields.io/badge/Made%20with-AI-333333?labelColor=f00) ![Verified by Humans](https://img.shields.io/badge/Verified%20by-Humans-333333?labelColor=brightgreen)

Local text-to-speech powered by Mistral's Voxtral model via MLX, with real-time streaming playback on Apple Silicon. Offers both an interactive CLI and a FastAPI server with an MCP bridge for AI agent integration.

### Features

| Feature | Description |
|---------|-------------|
| Streaming Playback | Real-time audio generation with lookahead — generates the next chunk while the current one plays |
| Interactive CLI | Terminal interface with model/voice selection, ESC to quit, and backspace support |
| REST API Server | FastAPI server with queued sequential playback and message status tracking |
| MCP Server | Ready-to-use MCP bridge for Claude Code and Claude Desktop integration |
| Multi-Voice | 20 voices across 9 languages (English, German, French, Spanish, Italian, Dutch, Portuguese, Hindi, Arabic) |
| Multi-Model | Supports 4-bit, 6-bit, and bf16 quantization variants of Voxtral 4B |

Under the hood, the project uses [mlx-audio](https://github.com/Blaizzy/mlx-audio) for model loading and inference on Apple Silicon, [sounddevice](https://python-sounddevice.readthedocs.io/) for real-time audio output, and [FastAPI](https://fastapi.tiangolo.com/) for the HTTP server. The MCP server is a lightweight TypeScript relay using the [Model Context Protocol SDK](https://modelcontextprotocol.io/).

## Design Principles

All configuration is explicit — no hardcoded defaults, no silent fallbacks. If a required value is missing from `config.yaml`, the application fails immediately with a clear error message. Audio files are saved to `data/output/` as WAV files with timestamps. The server uses a background worker thread with a lookahead pattern: it generates audio for the next request while the current one is still playing, eliminating gaps between consecutive messages.

## Prerequisites

- **Apple Silicon Mac** — MLX requires Apple Silicon (M1/M2/M3/M4)
- **Python 3.12+**
- **uv** — Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **just** — Command runner ([install](https://github.com/casey/just#installation))
- **Node.js 18+** — For the MCP server (optional)

## Project Structure

```
.
├── src/                    # Application source code
│   ├── main.py             # CLI frontend
│   ├── server.py           # FastAPI TTS server
│   └── tts.py              # Shared TTS engine (config, generation, playback)
├── tests/                  # Unit tests
│   ├── test_main.py
│   ├── test_server.py
│   ├── test_tts.py
│   └── architecture/       # Architecture import rule tests
├── scripts/                # Utility scripts
│   ├── download-model.sh   # Interactive model downloader
│   └── test-concurrent-say.py  # Concurrent /say load test
├── mcp/                    # MCP server (TypeScript)
│   └── tts-mcp.ts          # MCP relay to FastAPI server
├── config/                 # Static analysis rules
│   ├── semgrep/            # Semgrep custom rules
│   └── codespell/          # Spell-check configuration
├── data/
│   └── output/             # Generated WAV files
├── config.yaml             # Local configuration (gitignored)
├── justfile                # Command recipes
└── pyproject.toml          # Project metadata and dependencies
```

## Setup

```bash
just init
```

Creates report directories and installs all dependencies via `uv sync --all-extras`.

### Download a Model

```bash
just download
```

Presents three Voxtral 4B variants to choose from:

| Model | Size | Speed |
|-------|------|-------|
| `Voxtral-4B-TTS-2603-mlx-4bit` | ~2.5 GB | Fastest (RTF <1.0x) |
| `Voxtral-4B-TTS-2603-mlx-6bit` | ~3.5 GB | Balanced (RTF ~1.1x) |
| `Voxtral-4B-TTS-2603-mlx-bf16` | ~8.0 GB | Highest quality (RTF ~6.3x) |

After downloading, update `config.yaml` with the model path.

### Getting Started

1. Run `just init` — installs dependencies and prompts to download a model if none exists
2. Run `just download` — downloads a Voxtral model (if not already triggered by init)
3. Create `config.yaml` with the model path and server settings (see Configuration below)
4. Run `just serve` — starts the TTS server
5. Send requests via the API, CLI, or MCP bridge

## Configuration

All configuration lives in `config.yaml` at the project root. The file is gitignored and must be created manually. Example:

```yaml
model: /path/to/Voxtral-4B-TTS-2603-mlx-6bit
models_dir: /path/to/models
sample_rate: 24000
default_voice: casual_female
save_wav: true
simplify_punctuation: false
normalize_audio: true
target_lufs: -20.0
true_peak_ceiling_db: -1.0
min_duration_seconds: 0.5
host: 0.0.0.0
port: 12000
```

| Key | Description |
|-----|-------------|
| `model` | Path to the downloaded MLX model directory |
| `models_dir` | Base directory containing model subdirectories (for CLI model selection) |
| `sample_rate` | Audio sample rate in Hz (24000 for Voxtral) |
| `default_voice` | Default voice for server requests without a voice override |
| `save_wav` | Save generated audio to WAV files in `data/output/` (`true` or `false`) |
| `simplify_punctuation` | Strip commas, replace other marks with periods for cleaner speech |
| `normalize_audio` | Enable boost-only LUFS loudness normalization per utterance (`true` or `false`) |
| `target_lufs` | Target integrated loudness in LUFS when `normalize_audio` is enabled (e.g. `-20.0` for podcast mid) |
| `true_peak_ceiling_db` | Maximum allowed true peak in dBFS after gain is applied (e.g. `-1.0`); measured via 4x oversampling |
| `min_duration_seconds` | Utterances shorter than this are passed through unchanged (LUFS gating needs ~0.4s) |
| `host` | Server listen address |
| `port` | Server listen port |

### Loudness normalization

Different Voxtral voices produce audio at significantly different average levels. Enable `normalize_audio` to apply
utterance-level loudness normalization following ITU-R BS.1770-4 (the EBU R128 standard used in broadcast):

- The integrated loudness of each utterance is measured in LUFS using [`pyloudnorm`](https://github.com/csteinmetz1/pyloudnorm).
- If the measurement is below `target_lufs`, a single scalar gain is applied to lift the utterance toward the target.
- If the measurement is at or above the target, the audio is passed through unchanged — normalization is **boost-only
  and never attenuates**, so already-loud voices are preserved exactly.
- Before applying gain, the 4x-oversampled true peak is measured via `scipy.signal.resample_poly`. The gain is capped
  so the resulting true peak stays at or below `true_peak_ceiling_db`, preventing inter-sample clipping.
- Utterances shorter than `min_duration_seconds` and fully silent utterances are passed through unchanged.
- The same normalized audio is used for both speaker playback and the saved WAV file, so there is no drift between
  what you hear and what is written to disk.

## Usage

| Command | Description |
|---------|-------------|
| `just chat` | Start the interactive chat |
| `just serve` | Start the FastAPI TTS server (foreground) |
| `just stop` | Stop the running server |
| `just status` | Check if the server is running |

### CLI

```bash
just chat
```

Prompts for model and voice selection, then enters an interactive loop. Type text and press Enter twice to submit. Press ESC twice to quit.

For one-shot usage:

```bash
uv run -m src.main "Hello world" --voice casual_female
```

### Server

```bash
just serve
```

Starts a FastAPI server with queued playback. The server loads the model once at startup and processes requests sequentially through a background worker.

## API

FastAPI auto-generates interactive docs at `/docs` (Swagger) and `/redoc` (ReDoc) when the server is running.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/voices` | List available voices and default voice |
| POST | `/say` | Queue text for synthesis and playback (returns message ID) |
| GET | `/status/{message_id}` | Check status of a queued/playing/completed message |

### POST /say

```json
{
  "text": "Hello, this is a test.",
  "voice": "casual_female"
}
```

Returns `202 Accepted` with a message ID and queue position. Audio plays through the server's speakers.

### Message Lifecycle

`queued` -> `playing` -> `completed` (with audio file path) or `error` (with error details). Completed statuses expire after 1 hour.

## MCP Server

The MCP server (`mcp/tts-mcp.ts`) is a transparent relay between MCP clients and the FastAPI server. It exposes three tools:

| Tool | Description |
|------|-------------|
| `say` | Queue text for speech synthesis with a specified voice |
| `get_voices` | List all available voices |
| `get_status` | Check status of a speech request by message ID |

### Setup

```bash
cd mcp && npm install
```

### Usage with Claude Code / Claude Desktop

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "tts": {
      "command": "npx",
      "args": ["tsx", "tts-mcp.ts"],
      "cwd": "/path/to/mistral-text-to-spech/mcp"
    }
  }
}
```

The MCP server reads `config.yaml` from the project root to determine the server URL.

## Development

### Code Quality

| Command | Description |
|---------|-------------|
| `just code-format` | Auto-fix code style and formatting |
| `just code-style` | Check code style and formatting (read-only) |
| `just code-typecheck` | Run static type checking with mypy |
| `just code-lspchecks` | Run strict type checking with Pyright (LSP-based) |
| `just code-security` | Run security checks with bandit |
| `just code-deptry` | Check dependency hygiene with deptry |
| `just code-spell` | Check spelling in code and documentation |
| `just code-semgrep` | Run Semgrep static analysis |
| `just code-audit` | Scan dependencies for known vulnerabilities |
| `just code-architecture` | Run architecture import rule tests |
| `just code-stats` | Generate code statistics with pygount |

### Testing

| Command | Description |
|---------|-------------|
| `just test` | Run unit tests (fast) |
| `just test-coverage` | Run unit tests with coverage report (80% threshold) |

### CI

- `just ci` — Run all validation checks (verbose)
- `just ci-quiet` — Run all checks (silent, fail-fast)

The CI pipeline runs in order: init, code-format, code-style, code-typecheck, code-security, code-deptry, code-spell, code-semgrep, code-audit, test, code-architecture, code-lspchecks.

## AI-Assisted Development

This project includes a [CLAUDE.md](CLAUDE.md) file with development rules for AI coding assistants.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a full list of changes.

## Resources

- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — MLX-based audio models for Apple Silicon
- [Voxtral](https://mistral.ai/news/voxtral) — Mistral's text-to-speech model
- [Model Context Protocol](https://modelcontextprotocol.io/) — Open protocol for AI tool integration
- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
