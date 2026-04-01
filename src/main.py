"""CLI frontend for text-to-speech using Voxtral via mlx-audio."""

import argparse
import datetime
import queue
import sys
import termios
import threading
import tty
from pathlib import Path

from mlx_audio.tts.utils import load

from src.tts import (
    OUTPUT_DIR,
    audio_worker,
    clean_text,
    discover_models,
    discover_voices,
    load_config,
    make_output_path,
    simplify_punctuation,
)


def select_model(models: list[Path]) -> Path:
    """Display available models and let the user select one.

    Args:
        models: List of available model directory paths.

    Returns:
        Selected model directory path.
    """
    if len(models) == 1:
        print(f"\nUsing model: {models[0].name}")
        return models[0]

    print("\nAvailable models:")
    for i, model_path in enumerate(models, 1):
        print(f"  {i}. {model_path.name}")

    while True:
        choice = input(f"\nSelect model [1-{len(models)}]: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
        print(f"Invalid choice. Enter a number between 1 and {len(models)}.")


def select_voice(voices: list[str]) -> str:
    """Display available voices and let the user select one.

    Args:
        voices: List of available voice names.

    Returns:
        Selected voice name.
    """
    print("\nAvailable voices:")
    for i, voice in enumerate(voices, 1):
        print(f"  {i}. {voice}")

    while True:
        choice = input(f"\nSelect voice [1-{len(voices)}]: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(voices):
                return voices[idx]
        print(f"Invalid choice. Enter a number between 1 and {len(voices)}.")


def read_input(prompt: str) -> str | None:
    """Read a line of input, character by character. ESC exits, returns None.

    Args:
        prompt: The prompt to display.

    Returns:
        The entered text, or None if ESC was pressed.
    """
    sys.stdout.write(prompt)
    sys.stdout.flush()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    buf: list[str] = []

    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)

            if ch == "\x1b":  # ESC
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return None

            if ch in ("\r", "\n"):  # Enter
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return "".join(buf)

            if ch in ("\x7f", "\x08"):  # Backspace
                if buf:
                    buf.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue

            if ch == "\x03":  # Ctrl+C
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return None

            buf.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def list_outputs(output_dir: Path) -> None:
    """List previously generated audio files in the output directory.

    Args:
        output_dir: Directory containing generated WAV files.
    """
    if not output_dir.exists():
        print("No output directory found. No audio has been generated yet.")
        return

    wav_files = sorted(output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wav_files:
        print("No audio files found in output directory.")
        return

    print(f"\nGenerated audio files in {output_dir}:")
    for wav in wav_files:
        size_kb = wav.stat().st_size // 1024
        mtime = wav.stat().st_mtime
        ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {ts}  {wav.name}  ({size_kb} KB)")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Convert text to speech using Voxtral TTS via mlx-audio")
    parser.add_argument("text", nargs="?", help="Text to convert to speech (or enter interactively)")
    parser.add_argument("--model", help="MLX model path (overrides interactive selection)")
    parser.add_argument("--voice", help="Voice to use for synthesis (or select interactively)")
    parser.add_argument(
        "--list-outputs",
        action="store_true",
        help="List previously generated audio files and exit",
    )
    return parser


def resolve_model_dir(cli_model: str | None) -> str:
    """Resolve the model directory from CLI arg or interactive selection.

    Args:
        cli_model: Model path from CLI argument, or None.

    Returns:
        Resolved model directory path.

    Raises:
        ValueError: If no models_dir in config.yaml.
        FileNotFoundError: If the model directory does not exist.
    """
    if cli_model:
        if not Path(cli_model).exists():
            msg = f"Model directory does not exist: {cli_model}"
            raise FileNotFoundError(msg)
        return cli_model

    config = load_config()
    models_dir_str = config.get("models_dir")
    if not models_dir_str:
        msg = "No models_dir in config.yaml and no --model argument provided"
        raise ValueError(msg)

    models = discover_models(Path(models_dir_str))
    selected = select_model(models)
    return str(selected)


def prepare_text(text: str, simplify_punct: bool) -> str:
    """Clean text and optionally simplify punctuation.

    Args:
        text: Raw input text.
        simplify_punct: Whether punctuation simplification is enabled.

    Returns:
        Cleaned text, optionally with simplified punctuation.
    """
    prepared = clean_text(text)
    if prepared and simplify_punct:
        prepared = simplify_punctuation(prepared)
    return prepared


def shutdown_worker(work_queue: queue.Queue[str | None], worker: threading.Thread) -> None:
    """Wait for queued work to finish and stop the worker thread."""
    work_queue.join()
    work_queue.put(None)
    worker.join()


def main() -> None:
    """Main entry point for text-to-speech."""
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.list_outputs:
        list_outputs(OUTPUT_DIR)
        return

    config = load_config()
    raw_rate = config.get("sample_rate")
    if raw_rate is None:
        msg = "Missing required key 'sample_rate' in config.yaml"
        raise ValueError(msg)
    sample_rate = int(raw_rate)
    simplify_punct = bool(config.get("simplify_punctuation"))

    model_dir = resolve_model_dir(args.model)
    available_voices = discover_voices(Path(model_dir))

    voice: str = args.voice if args.voice else select_voice(available_voices)

    if voice not in available_voices:
        print(f"Error: voice '{voice}' not available. Choose from: {', '.join(available_voices)}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading model: {model_dir}")
    model = load(model_dir)

    if not hasattr(model, "generate") or model.generate is None:
        print(f"Error: model {model_dir} does not support generation", file=sys.stderr)
        sys.exit(1)

    print(f"Voice: {voice}")
    print("Type text and press Enter. Press ESC to quit.\n")

    output_path = make_output_path(OUTPUT_DIR)
    work_queue: queue.Queue[str | None] = queue.Queue()

    worker = threading.Thread(
        target=audio_worker,
        args=(work_queue, model, voice, output_path, sample_rate),
        daemon=True,
    )
    worker.start()

    if args.text:
        text = prepare_text(args.text, simplify_punct)
        if not text:
            print("Error: text is empty after cleaning", file=sys.stderr)
            sys.exit(1)
        work_queue.put(text)
        shutdown_worker(work_queue, worker)
        return

    while True:
        result = read_input("Text: ")
        if result is None:
            break
        text = prepare_text(result, simplify_punct)
        if not text:
            continue
        work_queue.put(text)
        print()

    shutdown_worker(work_queue, worker)


if __name__ == "__main__":
    main()
