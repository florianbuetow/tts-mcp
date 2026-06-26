"""Tests for CLI-specific functions in main module."""

from pathlib import Path

import pytest

from src.main import (
    create_argument_parser,
    list_outputs,
    load_cli_config,
    resolve_model_dir,
)


def _base_cli_config() -> dict[str, object]:
    return {
        "sample_rate": 24000,
        "save_wav": True,
        "simplify_punctuation": False,
        "normalize_audio": True,
        "target_lufs": -10.0,
        "true_peak_ceiling_db": -1.0,
        "min_duration_seconds": 0.5,
        "lead_silence_ms": 200,
    }


class TestLoadCliConfig:
    """Tests for CLI config parsing."""

    def test_loads_lead_silence_ms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.main.load_config", _base_cli_config)

        sample_rate, save_wav, simplify_punctuation, lead_silence_ms, normalization = load_cli_config()

        assert sample_rate == 24000
        assert save_wav is True
        assert simplify_punctuation is False
        assert lead_silence_ms == 200
        assert normalization.enabled is True

    def test_raises_when_lead_silence_ms_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _base_cli_config()
        del config["lead_silence_ms"]
        monkeypatch.setattr("src.main.load_config", lambda: config)

        with pytest.raises(ValueError, match="lead_silence_ms"):
            load_cli_config()


class TestResolveModelDir:
    """Tests for the resolve_model_dir function."""

    def test_uses_cli_arg(self, tmp_path: Path) -> None:
        model_path = str(tmp_path)
        result = resolve_model_dir(model_path)
        assert result == model_path

    def test_raises_if_cli_dir_missing(self):
        try:
            resolve_model_dir("/nonexistent/path")
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError:
            pass


class TestListOutputs:
    """Tests for the list_outputs function."""

    def test_prints_message_when_dir_missing(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        list_outputs(tmp_path / "nonexistent")

        out = capsys.readouterr().out
        assert "No output directory" in out

    def test_prints_message_when_no_wav_files(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        list_outputs(tmp_path)

        out = capsys.readouterr().out
        assert "No audio files" in out

    def test_lists_wav_files(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        (tmp_path / "speech_20260101_120000.wav").write_bytes(b"\x00" * 1000)
        (tmp_path / "speech_20260101_120030.wav").write_bytes(b"\x00" * 2000)

        list_outputs(tmp_path)

        out = capsys.readouterr().out
        assert "speech_20260101_120000.wav" in out
        assert "speech_20260101_120030.wav" in out


class TestCreateArgumentParser:
    """Tests for the create_argument_parser function."""

    def test_parses_text_argument(self):
        parser = create_argument_parser()
        args = parser.parse_args(["Hello world"])
        assert args.text == "Hello world"

    def test_parses_list_outputs_flag(self):
        parser = create_argument_parser()
        args = parser.parse_args(["--list-outputs"])
        assert args.list_outputs is True
