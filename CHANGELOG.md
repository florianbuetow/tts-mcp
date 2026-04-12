# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Renamed `just run` to `just chat` across justfile, README, and CLAUDE.md
- Interactive chat now requires pressing Enter twice to submit text, allowing multi-line input
- Interactive chat now requires pressing ESC twice to quit instead of once
- Empty enter no longer quits the interactive chat
- Improved `clean_text` input sanitization: tabs replaced with spaces, consecutive spaces and newlines collapsed independently without merging different whitespace types

### Added

- Utterance-level loudness normalization using ITU-R BS.1770-4 integrated LUFS measurement via `pyloudnorm`. Boost-only and asymmetric: quiet voices are lifted toward `target_lufs`, loud voices are left unchanged. Gain is capped by the 4x-oversampled true-peak measured with `scipy.signal.resample_poly` so the configured `true_peak_ceiling_db` is never exceeded. Controlled by four new `config.yaml` keys: `normalize_audio`, `target_lufs`, `true_peak_ceiling_db`, `min_duration_seconds`. A single `pyloudnorm.Meter` is constructed once per worker startup and reused for every utterance.
- Type stubs for `pyloudnorm` and `scipy.signal` under `stubs/` to satisfy pyright strict mode
- TTS engine core with streaming audio generation, playback, and WAV file saving using Voxtral models via mlx-audio
- Interactive CLI frontend with voice/model selection, raw terminal input, and background audio worker
- FastAPI TTS server with queued sequential playback, message status tracking, and automatic status eviction
- MCP server bridge for AI agent integration via Model Context Protocol
- `save_wav` config parameter to toggle WAV file saving on/off without impacting playback
- Interactive model download script with support for 4-bit, 6-bit, and bf16 quantizations
- `just download` target for manual model downloads; `just init` auto-triggers download when no model exists
- Justfile with build, run, serve, stop, status, and comprehensive CI recipes
- Unit tests and architecture import rule tests with 80% coverage threshold
- Load testing utility script for server benchmarking
- Application config with linter rules, static analysis (ruff, mypy, pyright, bandit, semgrep, deptry, codespell), and security scanning
- Type stubs for mlx_audio and sounddevice
- Project documentation (README, CLAUDE.md, QUICKSTART)
- MIT license, gitignore, and data directory scaffold

[Unreleased]: https://github.com/florianbuetow/tts-mcp/commits/main
