"""FastAPI TTS server with queued sequential playback."""

import contextlib
import dataclasses
import datetime
import json
import logging
import queue
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.tts import (
    OUTPUT_DIR,
    TTSModel,
    clean_text,
    discover_voices,
    generate_chunks,
    load_config,
    make_output_path,
    play_chunks,
    simplify_punctuation,
)

logger = logging.getLogger("tts-server")

STATUS_TTL_SECONDS: int = 3600


@dataclasses.dataclass
class WorkItem:
    """A queued work item for the audio worker."""

    message_id: str
    text: str
    voice: str


@dataclasses.dataclass
class MessageStatus:
    """Tracks the lifecycle of a queued message."""

    message_id: str
    status: str
    text: str
    audio_file: str | None
    error: str | None
    completed_at: float | None


class ServerState:
    """Mutable server state shared between endpoints and the audio worker."""

    def __init__(
        self,
        model: TTSModel,
        voices: list[str],
        default_voice: str,
        sample_rate: int,
        simplify_punctuation: bool,
    ) -> None:
        """Initialize server state.

        Args:
            model: Loaded TTS model.
            voices: Available voice names.
            default_voice: Default voice for requests without voice override.
            sample_rate: Audio sample rate in Hz.
            simplify_punctuation: Whether to simplify punctuation before TTS.
        """
        self.model = model
        self.voices = voices
        self.default_voice = default_voice
        self.sample_rate = sample_rate
        self.simplify_punctuation = simplify_punctuation
        self.work_queue: queue.Queue[WorkItem | None] = queue.Queue()
        self.statuses: dict[str, MessageStatus] = {}
        self.status_lock = threading.Lock()
        self._counter = 0
        self._counter_lock = threading.Lock()

    def next_message_id(self) -> str:
        """Generate a unique message ID.

        Returns:
            Message ID in format msg_YYYYMMDD_HHMMSS_NNN.
        """
        with self._counter_lock:
            self._counter += 1
            counter = self._counter
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"msg_{ts}_{counter:03d}"

    def evict_expired(self) -> None:
        """Remove completed/errored status entries older than TTL."""
        now = time.time()
        with self.status_lock:
            expired = [
                mid for mid, ms in self.statuses.items() if ms.completed_at is not None and (now - ms.completed_at) > STATUS_TTL_SECONDS
            ]
            for mid in expired:
                del self.statuses[mid]


class SayRequest(BaseModel):
    """Request body for POST /say."""

    text: str
    voice: str | None = None


class SayResponse(BaseModel):
    """Response body for POST /say."""

    message_id: str
    status: str
    queue_position: int


class StatusResponse(BaseModel):
    """Response body for GET /status/{message_id}."""

    message_id: str
    status: str
    text: str
    audio_file: str | None
    error: str | None


class VoicesResponse(BaseModel):
    """Response body for GET /voices."""

    voices: list[str]
    default_voice: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str


router = APIRouter()


@router.get("/health")
def health() -> HealthResponse:
    """Liveness check."""
    return HealthResponse(status="ok")


@router.get("/voices")
def voices(request: Request) -> VoicesResponse:
    """List available voices."""
    state: ServerState = request.app.state.server
    return VoicesResponse(voices=state.voices, default_voice=state.default_voice)


@router.post("/say", status_code=202)
def say(request: Request, body: SayRequest) -> SayResponse:
    """Queue text for speech synthesis and playback."""
    state: ServerState = request.app.state.server

    cleaned = clean_text(body.text)
    if not cleaned:
        raise HTTPException(status_code=422, detail="Text is empty after cleaning")

    if state.simplify_punctuation:
        cleaned = simplify_punctuation(cleaned)

    voice = body.voice if body.voice else state.default_voice
    if voice not in state.voices:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{voice}' not available. Available voices: {', '.join(state.voices)}",
        )

    message_id = state.next_message_id()
    state.evict_expired()

    queue_position = state.work_queue.qsize()
    with state.status_lock:
        state.statuses[message_id] = MessageStatus(
            message_id=message_id,
            status="queued",
            text=cleaned,
            audio_file=None,
            error=None,
            completed_at=None,
        )

    state.work_queue.put(WorkItem(message_id=message_id, text=cleaned, voice=voice))

    logger.debug(
        "POST /say request:\n%s",
        json.dumps({"text": body.text, "voice": voice, "message_id": message_id}, indent=2),
    )

    return SayResponse(message_id=message_id, status="queued", queue_position=queue_position)


@router.get("/status/{message_id}")
def status(request: Request, message_id: str) -> StatusResponse:
    """Check the status of a queued/playing/completed message."""
    state: ServerState = request.app.state.server

    state.evict_expired()

    with state.status_lock:
        ms = state.statuses.get(message_id)

    if ms is None:
        raise HTTPException(status_code=404, detail=f"Unknown message ID: {message_id}")

    return StatusResponse(
        message_id=ms.message_id,
        status=ms.status,
        text=ms.text,
        audio_file=ms.audio_file,
        error=ms.error,
    )


def _finish_playback(
    state: ServerState,
    message_id: str,
    chunks: list[np.ndarray],
    output_path: Path,
) -> None:
    """Play audio chunks and update message status to completed or error.

    Args:
        state: Server state with status dict and lock.
        message_id: ID of the message being played.
        chunks: Audio chunks to play.
        output_path: Path to save the WAV file.
    """
    try:
        play_chunks(chunks, output_path, state.sample_rate)
        with state.status_lock:
            ms = state.statuses[message_id]
            ms.status = "completed"
            ms.audio_file = str(output_path)
            ms.completed_at = time.time()
        logger.debug("Playback completed for %s -> %s", message_id, output_path)
    except Exception as exc:
        logger.error("Playback failed for %s: %s", message_id, exc)
        with state.status_lock:
            ms = state.statuses[message_id]
            ms.status = "error"
            ms.error = str(exc)
            ms.completed_at = time.time()


def _fail_item(state: ServerState, message_id: str, error: str) -> None:
    """Mark a work item as failed in the status dict.

    Args:
        state: Server state with status dict and lock.
        message_id: ID of the failed message.
        error: Error description.
    """
    with state.status_lock:
        ms = state.statuses.get(message_id)
        if ms is not None:
            ms.status = "error"
            ms.error = error
            ms.completed_at = time.time()


def _start_playback(
    state: ServerState,
    pending: tuple[WorkItem, list[np.ndarray]],
    playback_thread: threading.Thread | None,
) -> threading.Thread:
    """Wait for any prior playback, then start a new playback thread.

    Args:
        state: Server state with status dict and lock.
        pending: Tuple of (work_item, audio_chunks) ready for playback.
        playback_thread: Previously running playback thread, or None.

    Returns:
        Newly started playback thread.
    """
    if playback_thread is not None:
        playback_thread.join()

    work_item, chunks = pending

    with state.status_lock:
        state.statuses[work_item.message_id].status = "playing"

    output_path = make_output_path(OUTPUT_DIR)
    thread = threading.Thread(
        target=_finish_playback,
        args=(state, work_item.message_id, chunks, output_path),
        daemon=True,
    )
    thread.start()
    return thread


def server_audio_worker(state: ServerState) -> None:
    """Background worker that processes queued TTS requests sequentially.

    Uses lookahead pattern: generates chunks for the next message while
    the current one is still playing.  Wraps each iteration in a top-level
    handler so that unexpected exceptions are logged and the worker keeps
    running instead of dying silently.

    Args:
        state: Server state with work queue, model, and status tracking.
    """
    pending: tuple[WorkItem, list[np.ndarray]] | None = None
    playback_thread: threading.Thread | None = None

    while True:
        current_item: WorkItem | None = None
        try:
            if pending is not None:
                playback_thread = _start_playback(state, pending, playback_thread)
                pending = None

            item = state.work_queue.get()
            if item is None:
                if playback_thread is not None:
                    playback_thread.join()
                break

            current_item = item

            try:
                chunks = generate_chunks(state.model, item.text, item.voice)
                pending = (item, chunks)
            except (RuntimeError, ValueError) as exc:
                logger.error("TTS generation failed for %s: %s", item.message_id, exc)
                _fail_item(state, item.message_id, str(exc))

            state.work_queue.task_done()

        except Exception as exc:
            logger.error(
                "Audio worker caught unexpected error (recovering): %s",
                exc,
                exc_info=True,
            )
            if current_item is not None:
                _fail_item(state, current_item.message_id, f"unexpected worker error: {exc}")
                with contextlib.suppress(ValueError):
                    state.work_queue.task_done()

    if pending is not None:
        _start_playback(state, pending, playback_thread)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load model on startup, shut down worker on exit."""
    from mlx_audio.tts.utils import load

    config = load_config()

    model_path = config.get("model")
    if not model_path:
        msg = "Missing required key 'model' in config.yaml"
        raise ValueError(msg)
    if not Path(model_path).exists():
        msg = f"Model directory does not exist: {model_path}"
        raise FileNotFoundError(msg)

    raw_rate = config.get("sample_rate")
    if raw_rate is None:
        msg = "Missing required key 'sample_rate' in config.yaml"
        raise ValueError(msg)
    sample_rate = int(raw_rate)

    default_voice = config.get("default_voice")
    if not default_voice:
        msg = "Missing required key 'default_voice' in config.yaml"
        raise ValueError(msg)

    simplify_punct = bool(config.get("simplify_punctuation"))

    available_voices = discover_voices(Path(model_path))
    if default_voice not in available_voices:
        msg = f"default_voice '{default_voice}' not found. Available: {', '.join(available_voices)}"
        raise ValueError(msg)

    model = load(model_path)

    if not hasattr(model, "generate") or model.generate is None:
        msg = f"Model {model_path} does not support generation"
        raise RuntimeError(msg)

    state = ServerState(
        model=cast(TTSModel, model),
        voices=available_voices,
        default_voice=default_voice,
        sample_rate=sample_rate,
        simplify_punctuation=simplify_punct,
    )

    worker = threading.Thread(target=server_audio_worker, args=(state,), daemon=True)
    worker.start()

    app.state.server = state

    yield

    state.work_queue.put(None)
    worker.join(timeout=10)


app = FastAPI(lifespan=lifespan)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    _config = load_config()

    _host = _config.get("host")
    if not _host:
        _msg = "Missing required key 'host' in config.yaml"
        raise ValueError(_msg)

    _raw_port = _config.get("port")
    if _raw_port is None:
        _msg = "Missing required key 'port' in config.yaml"
        raise ValueError(_msg)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    uvicorn.run("src.server:app", host=_host, port=int(_raw_port), log_level="debug")
