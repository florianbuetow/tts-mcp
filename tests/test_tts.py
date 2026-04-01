"""Tests for the shared TTS engine."""

import queue
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.tts import (
    audio_worker,
    clean_text,
    discover_models,
    discover_voices,
    generate_chunks,
    generate_speech,
    load_config,
    make_output_path,
    play_audio,
    play_chunks,
    save_audio,
    simplify_punctuation,
)


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_raises_if_config_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.tts.CONFIG_PATH", tmp_path / "nonexistent.yaml")
        try:
            load_config()
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError:
            pass

    def test_loads_valid_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("models_dir: /some/path\n")
        monkeypatch.setattr("src.tts.CONFIG_PATH", config_file)

        config = load_config()
        assert config["models_dir"] == "/some/path"


class TestDiscoverModels:
    """Tests for the discover_models function."""

    def test_discovers_models(self, tmp_path: Path) -> None:
        model_a = tmp_path / "model-a"
        model_a.mkdir()
        (model_a / "model.safetensors").write_bytes(b"fake")

        model_b = tmp_path / "model-b"
        model_b.mkdir()
        (model_b / "model.safetensors").write_bytes(b"fake")

        models = discover_models(tmp_path)
        names = [m.name for m in models]
        assert "model-a" in names
        assert "model-b" in names

    def test_raises_if_dir_missing(self, tmp_path: Path) -> None:
        try:
            discover_models(tmp_path / "nonexistent")
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError:
            pass

    def test_raises_if_no_models(self, tmp_path: Path) -> None:
        try:
            discover_models(tmp_path)
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError:
            pass


class TestDiscoverVoices:
    """Tests for the discover_voices function."""

    def test_discovers_voices(self, tmp_path: Path) -> None:
        voice_dir = tmp_path / "voice_embedding"
        voice_dir.mkdir()
        (voice_dir / "casual_male.safetensors").write_bytes(b"fake")
        (voice_dir / "neutral_female.safetensors").write_bytes(b"fake")

        voices = discover_voices(tmp_path)
        assert voices == ["casual_male", "neutral_female"]

    def test_raises_if_no_voice_dir(self, tmp_path: Path) -> None:
        try:
            discover_voices(tmp_path)
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError:
            pass

    def test_raises_if_no_voices(self, tmp_path: Path) -> None:
        (tmp_path / "voice_embedding").mkdir()
        try:
            discover_voices(tmp_path)
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError:
            pass


class TestGenerateSpeech:
    """Tests for the generate_speech function."""

    @patch("src.tts.load")
    def test_generates_audio_from_text(self, mock_load: MagicMock) -> None:
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.audio = np.ones(1000, dtype=np.float32)
        mock_model.generate.return_value = [mock_result]
        mock_load.return_value = mock_model

        audio = generate_speech(model_id="test-model", text="Hello", voice="casual_male")

        assert len(audio) == 1000
        mock_load.assert_called_once_with("test-model")
        mock_model.generate.assert_called_once_with(text="Hello", voice="casual_male")

    @patch("src.tts.load")
    def test_concatenates_multiple_chunks(self, mock_load: MagicMock) -> None:
        mock_model = MagicMock()
        chunk1 = MagicMock()
        chunk1.audio = np.ones(500, dtype=np.float32)
        chunk2 = MagicMock()
        chunk2.audio = np.ones(300, dtype=np.float32)
        mock_model.generate.return_value = [chunk1, chunk2]
        mock_load.return_value = mock_model

        audio = generate_speech(model_id="m", text="test", voice="neutral_male")

        assert len(audio) == 800

    @patch("src.tts.load")
    def test_raises_if_no_audio_generated(self, mock_load: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.generate.return_value = []
        mock_load.return_value = mock_model

        try:
            generate_speech(model_id="m", text="test", voice="casual_male")
            raise AssertionError("Expected RuntimeError")
        except RuntimeError as exc:
            assert "No audio was generated" in str(exc)


class TestPlayAudio:
    """Tests for the play_audio function."""

    @patch("src.tts.sd")
    def test_plays_audio_at_sample_rate(self, mock_sd: MagicMock) -> None:
        audio = np.zeros(1000, dtype=np.float32)
        play_audio(audio, 24000)

        mock_sd.play.assert_called_once()
        call_args = mock_sd.play.call_args
        np.testing.assert_array_equal(call_args[0][0], audio)
        assert call_args[0][1] == 24000
        mock_sd.wait.assert_called_once()


class TestSaveAudio:
    """Tests for the save_audio function."""

    def test_saves_wav_file(self, tmp_path: Path) -> None:
        audio = np.zeros(1000, dtype=np.float32)
        output_path = tmp_path / "test.wav"

        save_audio(audio, output_path, sample_rate=24000)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        audio = np.zeros(1000, dtype=np.float32)
        output_path = tmp_path / "nested" / "dir" / "test.wav"

        save_audio(audio, output_path, sample_rate=24000)

        assert output_path.exists()


class TestGenerateChunks:
    """Tests for the generate_chunks function."""

    def test_returns_list_of_arrays(self):
        mock_model = MagicMock()
        chunk = MagicMock()
        chunk.audio = np.ones(500, dtype=np.float32)
        mock_model.generate.return_value = [chunk]

        result = generate_chunks(mock_model, "hello", "casual_female")

        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)
        assert len(result[0]) == 500

    def test_returns_empty_list_when_model_yields_nothing(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = []

        result = generate_chunks(mock_model, "hello", "casual_female")

        assert result == []

    def test_passes_text_and_voice_to_model(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = []

        generate_chunks(mock_model, "test text", "neutral_male")

        mock_model.generate.assert_called_once_with(text="test text", voice="neutral_male")


class TestPlayChunks:
    """Tests for the play_chunks function."""

    @patch("src.tts.sd")
    def test_streams_chunks_and_saves_file(self, mock_sd: MagicMock, tmp_path: Path) -> None:
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value.__enter__ = MagicMock(return_value=mock_stream)
        mock_sd.OutputStream.return_value.__exit__ = MagicMock(return_value=False)

        chunks = [np.ones(100, dtype=np.float32), np.ones(200, dtype=np.float32)]
        output_path = tmp_path / "out.wav"

        play_chunks(chunks, output_path, sample_rate=24000)

        assert mock_stream.write.call_count == 2
        assert output_path.exists()


class TestAudioWorker:
    """Tests for the audio_worker function."""

    @patch("src.tts.play_chunks")
    def test_processes_text_and_signals_done(self, mock_play: MagicMock, tmp_path: Path) -> None:
        mock_model = MagicMock()
        chunk = MagicMock()
        chunk.audio = np.ones(100, dtype=np.float32)
        mock_model.generate.return_value = [chunk]

        work_queue: queue.Queue[str | None] = queue.Queue()
        work_queue.put("hello world")
        work_queue.put(None)

        t = threading.Thread(
            target=audio_worker,
            args=(work_queue, mock_model, "casual_female", tmp_path / "out.wav", 24000),
        )
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        mock_model.generate.assert_called_once_with(text="hello world", voice="casual_female")

    @patch("src.tts.play_chunks")
    def test_processes_multiple_items(self, mock_play: MagicMock, tmp_path: Path) -> None:
        mock_model = MagicMock()
        chunk = MagicMock()
        chunk.audio = np.ones(100, dtype=np.float32)
        mock_model.generate.return_value = [chunk]

        work_queue: queue.Queue[str | None] = queue.Queue()
        work_queue.put("first")
        work_queue.put("second")
        work_queue.put(None)

        t = threading.Thread(
            target=audio_worker,
            args=(work_queue, mock_model, "neutral_male", tmp_path / "out.wav", 24000),
        )
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        assert mock_model.generate.call_count == 2

    @patch("src.tts.play_chunks")
    def test_shuts_down_on_none_sentinel(self, mock_play: MagicMock, tmp_path: Path) -> None:
        mock_model = MagicMock()

        work_queue: queue.Queue[str | None] = queue.Queue()
        work_queue.put(None)

        t = threading.Thread(
            target=audio_worker,
            args=(work_queue, mock_model, "casual_female", tmp_path / "out.wav", 24000),
        )
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        mock_model.generate.assert_not_called()


class TestMakeOutputPath:
    """Tests for the make_output_path function."""

    def test_returns_path_in_output_dir(self, tmp_path: Path) -> None:
        result = make_output_path(tmp_path)

        assert result.parent == tmp_path
        assert result.name.startswith("speech_")
        assert result.suffix == ".wav"

    def test_includes_timestamp(self, tmp_path: Path) -> None:
        result = make_output_path(tmp_path)

        parts = result.stem.split("_")
        assert len(parts) == 3
        assert parts[0] == "speech"
        assert len(parts[1]) == 8
        assert len(parts[2]) == 6


class TestCleanText:
    """Tests for the clean_text function."""

    def test_strips_and_collapses_spaces(self):
        assert clean_text("  hello   world  ") == "hello world"

    def test_collapses_newlines(self):
        assert clean_text("line1\n\n\nline2") == "line1\nline2"

    def test_whitespace_only_returns_empty(self):
        assert clean_text("   ") == ""

    def test_already_clean_unchanged(self):
        assert clean_text("hello") == "hello"

    def test_tabs_collapsed_to_space(self):
        assert clean_text("\t\thello\t\tworld") == "hello world"

    def test_mixed_whitespace(self):
        assert clean_text("  hello \t world \n\n next  ") == "hello world\nnext"


class TestSimplifyPunctuation:
    """Tests for the simplify_punctuation function."""

    def test_comma_removed_exclamation_to_period(self):
        assert simplify_punctuation("Hello, world!") == "Hello world."

    def test_ellipsis_and_question_mark(self):
        assert simplify_punctuation("Wait... what?") == "Wait. what."

    def test_brackets_to_periods(self):
        assert simplify_punctuation("(yes) [no] {maybe}") == "yes. no. maybe."

    def test_quotes_and_comma(self):
        assert simplify_punctuation('He said, "really?"') == "He said. really."

    def test_colon_to_period_preserves_decimal(self):
        assert simplify_punctuation("Price: $5.00") == "Price. $5.00"

    def test_em_dash_to_period(self):
        assert simplify_punctuation("Hello—world") == "Hello. world"

    def test_periods_with_spaces_collapsed(self):
        assert simplify_punctuation("A. . .B") == "A. B"

    def test_already_simplified_unchanged(self):
        assert simplify_punctuation("No change needed.") == "No change needed."

    def test_fullwidth_comma_removed(self):
        assert simplify_punctuation("Hello， world") == "Hello world"

    def test_smart_quotes_to_period(self):
        assert simplify_punctuation("“Hello”") == "Hello."

    def test_en_dash_to_period(self):
        assert simplify_punctuation("A–B") == "A. B"

    def test_ellipsis_character(self):
        assert simplify_punctuation("Wait… what") == "Wait. what"

    def test_semicolon_to_period(self):
        assert simplify_punctuation("first; second") == "first. second"
