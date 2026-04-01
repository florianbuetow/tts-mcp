"""Shared TTS engine: config, model/voice discovery, audio generation, playback, and saving."""

import datetime
import queue
import re
import sys
import threading
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import sounddevice as sd
import yaml
from mlx_audio.tts.utils import load

OUTPUT_DIR = Path("data/output")
CONFIG_PATH = Path("config.yaml")


class GenerationResult(Protocol):
    """Protocol for a single TTS generation result chunk."""

    @property
    def audio(self) -> np.ndarray:
        """Audio samples for this chunk."""
        ...


class TTSModel(Protocol):
    """Protocol for a TTS model that supports streaming generation."""

    def generate(self, text: str, voice: str) -> Iterator[GenerationResult]:
        """Generate speech audio chunks from text."""
        ...


def clean_text(text: str) -> str:
    """Clean text by stripping and collapsing whitespace.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text. Empty string if input was only whitespace.
    """
    text = text.strip()
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" *\n *", "\n", text)
    text = text.strip()
    return text


def simplify_punctuation(text: str) -> str:
    """Simplify punctuation by removing commas and replacing other marks with periods.

    Handles ASCII punctuation, smart quotes, em/en dashes, and ellipsis.
    CJK and other script-specific punctuation is passed through unchanged.

    Args:
        text: Input text (should be pre-cleaned with clean_text).

    Returns:
        Text with simplified punctuation.
    """
    text = text.replace(",", "")
    text = text.replace("\uff0c", "")

    text = text.replace("...", ".")
    text = text.replace("--", ".")

    for ch in "!?;:()[]{}\"'`\u2014\u2013\u2026\u201c\u201d\u2018\u2019":
        text = text.replace(ch, ".")

    text = re.sub(r"\.\s*(?:\.\s*)+", ".", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\.(?=[^\s.\d])", ". ", text)
    text = re.sub(r"^[\s.]+", "", text)
    text = text.rstrip()

    return text


def load_config() -> dict[str, Any]:
    """Load configuration from config.yaml.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config.yaml does not exist.
        ValueError: If config.yaml is empty or invalid.
    """
    if not CONFIG_PATH.exists():
        msg = f"Configuration file not found: {CONFIG_PATH}"
        raise FileNotFoundError(msg)

    with CONFIG_PATH.open() as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        msg = f"Invalid config.yaml: expected a mapping, got {type(raw).__name__}"
        raise ValueError(msg)

    return cast(dict[str, Any], raw)


def discover_models(models_dir: Path) -> list[Path]:
    """Discover downloaded TTS models in the models directory.

    Args:
        models_dir: Base directory containing model subdirectories.

    Returns:
        Sorted list of model directory paths that contain model.safetensors.

    Raises:
        FileNotFoundError: If models_dir does not exist or has no models.
    """
    if not models_dir.exists():
        msg = f"Models directory does not exist: {models_dir}"
        raise FileNotFoundError(msg)

    models = sorted(p.parent for p in models_dir.glob("*/model.safetensors"))
    if not models:
        msg = f"No models found in {models_dir}. Run ./scripts/download-model.sh first."
        raise FileNotFoundError(msg)

    return models


def discover_voices(model_dir: Path) -> list[str]:
    """Discover available voices from the model's voice_embedding directory.

    Args:
        model_dir: Path to the model directory.

    Returns:
        Sorted list of available voice names.

    Raises:
        FileNotFoundError: If voice_embedding directory does not exist or has no voices.
    """
    voice_dir = model_dir / "voice_embedding"
    if not voice_dir.exists():
        msg = f"No voice_embedding directory found in {model_dir}"
        raise FileNotFoundError(msg)

    voices = sorted(p.stem for p in voice_dir.glob("*.safetensors"))
    if not voices:
        msg = f"No voice files found in {voice_dir}"
        raise FileNotFoundError(msg)

    return voices


def generate_speech(model_id: str, text: str, voice: str) -> np.ndarray:
    """Generate speech audio from text using Voxtral TTS.

    Args:
        model_id: The MLX model identifier to load.
        text: The text to convert to speech.
        voice: The voice to use for synthesis.

    Returns:
        Audio samples as a numpy array at 24kHz.

    Raises:
        RuntimeError: If no audio was generated.
    """
    model = load(model_id)

    if not hasattr(model, "generate") or model.generate is None:
        msg = f"Model {model_id} does not support generation"
        raise RuntimeError(msg)

    audio_chunks: list[np.ndarray] = []
    for result in model.generate(text=text, voice=voice):
        chunk = np.array(result.audio)
        audio_chunks.append(chunk)

    if not audio_chunks:
        msg = "No audio was generated by the model"
        raise RuntimeError(msg)

    return np.concatenate(audio_chunks)


def play_audio(audio: np.ndarray, sample_rate: int) -> None:
    """Play audio samples through the default audio device.

    Args:
        audio: Audio samples as a numpy array.
        sample_rate: Sample rate in Hz.
    """
    sd.play(audio, sample_rate)
    sd.wait()


def save_audio(audio: np.ndarray, output_path: Path, sample_rate: int) -> None:
    """Save audio samples to a WAV file.

    Args:
        audio: Audio samples as a numpy array.
        output_path: Path to save the WAV file.
        sample_rate: Sample rate in Hz.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def generate_chunks(model: TTSModel, text: str, voice: str) -> list[np.ndarray]:
    """Generate audio chunks from text without playing.

    Args:
        model: Loaded TTS model.
        text: The text to convert to speech.
        voice: The voice to use for synthesis.

    Returns:
        List of audio chunks as numpy arrays.
    """
    return [np.array(result.audio, dtype=np.float32) for result in model.generate(text=text, voice=voice)]


def play_chunks(chunks: list[np.ndarray], output_path: Path | None, sample_rate: int) -> None:
    """Stream audio chunks to speakers and optionally save to file.

    Args:
        chunks: List of audio chunks as numpy arrays.
        output_path: Path to save the generated WAV file, or None to skip saving.
        sample_rate: Sample rate in Hz.
    """
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        for chunk in chunks:
            stream.write(chunk.reshape(-1, 1))

    if output_path is not None:
        audio = np.concatenate(chunks)
        save_audio(audio, output_path, sample_rate)


def audio_worker(
    work_queue: queue.Queue[str | None],
    model: TTSModel,
    voice: str,
    output_path: Path | None,
    sample_rate: int,
) -> None:
    """Background worker that generates and plays TTS audio.

    Generates audio for the next text while the current one is still playing,
    so there is no gap between sentences.

    Args:
        work_queue: Queue of text strings to synthesize. None signals shutdown.
        model: Loaded TTS model.
        voice: Voice to use for synthesis.
        output_path: Path to save generated audio, or None to skip saving.
        sample_rate: Sample rate in Hz.
    """
    pending_chunks: list[np.ndarray] | None = None
    playback_thread: threading.Thread | None = None

    while True:
        if pending_chunks is not None:
            if playback_thread is not None:
                playback_thread.join()
            chunks_to_play = pending_chunks
            pending_chunks = None
            playback_thread = threading.Thread(
                target=play_chunks,
                args=(chunks_to_play, output_path, sample_rate),
                daemon=True,
            )
            playback_thread.start()

        text = work_queue.get()
        if text is None:
            if playback_thread is not None:
                playback_thread.join()
            break

        try:
            pending_chunks = generate_chunks(model, text, voice)
        except (RuntimeError, ValueError) as exc:
            print(f"\n  Error: {exc}", file=sys.stderr)
        work_queue.task_done()

    if pending_chunks is not None:
        if playback_thread is not None:
            playback_thread.join()
        play_chunks(pending_chunks, output_path, sample_rate)


def make_output_path(output_dir: Path) -> Path:
    """Generate a timestamped output path for a new audio file.

    Args:
        output_dir: Directory to save audio files.

    Returns:
        Path with a timestamp-based filename.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"speech_{ts}.wav"
