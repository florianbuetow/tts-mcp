"""Tests for the FastAPI TTS server."""

import re
import threading
import time
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pyloudnorm as pyln
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.server import (
    STATUS_TTL_SECONDS,
    MessageStatus,
    ServerState,
    WorkItem,
    router,
    server_audio_worker,
)
from src.tts import AudioPlayer


def _make_state(
    voices: list[str] | None = None,
    default_voice: str = "casual_female",
    sample_rate: int = 24000,
    simplify_punctuation: bool = False,
    save_wav: bool = True,
    lead_silence_ms: int = 200,
    normalize_audio: bool = False,
    target_lufs: float = -20.0,
    true_peak_ceiling_db: float = -1.0,
    min_duration_seconds: float = 0.5,
    meter: pyln.Meter | None = None,
    preload_model: bool = True,
    model_path: str = "test-model-path",
) -> ServerState:
    """Create a ServerState for testing.

    By default a mock model is pre-loaded (preload_model=True). Pass
    preload_model=False to leave state.model as None so the worker loads it
    on its own thread, mirroring the production code path.
    """
    model = MagicMock() if preload_model else None
    if voices is None:
        voices = ["casual_female", "casual_male"]
    if meter is None:
        meter = pyln.Meter(float(sample_rate))
    return ServerState(
        model=model,
        model_path=model_path,
        voices=voices,
        default_voice=default_voice,
        sample_rate=sample_rate,
        lead_silence_ms=lead_silence_ms,
        simplify_punctuation=simplify_punctuation,
        save_wav=save_wav,
        normalize_audio=normalize_audio,
        target_lufs=target_lufs,
        true_peak_ceiling_db=true_peak_ceiling_db,
        min_duration_seconds=min_duration_seconds,
        meter=meter,
    )


def _make_app(state: ServerState) -> FastAPI:
    """Create a test FastAPI app with the given state."""
    app = FastAPI()
    app.state.server = state
    app.include_router(router)
    return app


class _ImmediateAudioPlayer:
    """Synchronous fake for server worker tests."""

    playback_error: Exception | None = None
    active_count = 0
    max_active_count = 0

    def __init__(self, sample_rate: int, lead_silence_ms: int) -> None:
        self.sample_rate = sample_rate
        self.lead_silence_ms = lead_silence_ms

    def submit(self, job: Any) -> None:
        _ImmediateAudioPlayer.active_count += 1
        _ImmediateAudioPlayer.max_active_count = max(
            _ImmediateAudioPlayer.max_active_count,
            _ImmediateAudioPlayer.active_count,
        )
        try:
            if _ImmediateAudioPlayer.playback_error is not None:
                if job.on_error is not None:
                    job.on_error(_ImmediateAudioPlayer.playback_error)
                return
            if job.on_complete is not None:
                job.on_complete(job.output_path)
        finally:
            _ImmediateAudioPlayer.active_count -= 1

    def close(self) -> None:
        return


@pytest.fixture(autouse=True)
def _use_immediate_audio_player(monkeypatch: pytest.MonkeyPatch) -> None:
    _ImmediateAudioPlayer.playback_error = None
    _ImmediateAudioPlayer.active_count = 0
    _ImmediateAudioPlayer.max_active_count = 0
    monkeypatch.setattr("src.server.AudioPlayer", _ImmediateAudioPlayer)


class TestHealth:
    """Tests for GET /health."""

    def test_returns_ok(self):
        state = _make_state()
        app = _make_app(state)
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestVoices:
    """Tests for GET /voices."""

    def test_returns_voices_and_default(self):
        state = _make_state(voices=["casual_female", "neutral_male"], default_voice="casual_female")
        app = _make_app(state)
        client = TestClient(app)

        response = client.get("/voices")

        assert response.status_code == 200
        data = response.json()
        assert data["voices"] == ["casual_female", "neutral_male"]
        assert data["default_voice"] == "casual_female"


class TestMessageId:
    """Tests for message ID generation."""

    def test_format_matches_pattern(self):
        state = _make_state()
        msg_id = state.next_message_id()
        assert re.match(r"^msg_\d{8}_\d{6}_\d{3}$", msg_id)

    def test_counter_increments(self):
        state = _make_state()
        id1 = state.next_message_id()
        id2 = state.next_message_id()
        c1 = int(id1.rsplit("_", 1)[1])
        c2 = int(id2.rsplit("_", 1)[1])
        assert c2 == c1 + 1

    def test_ids_are_unique(self):
        state = _make_state()
        ids = {state.next_message_id() for _ in range(100)}
        assert len(ids) == 100


class TestSay:
    """Tests for POST /say."""

    def test_valid_text_returns_202(self):
        state = _make_state()
        app = _make_app(state)
        client = TestClient(app)

        response = client.post("/say", json={"text": "Hello world"})

        assert response.status_code == 202
        data = response.json()
        assert "message_id" in data
        assert data["status"] == "queued"
        assert data["queue_position"] >= 0

    def test_empty_text_returns_422(self):
        state = _make_state()
        app = _make_app(state)
        client = TestClient(app)

        response = client.post("/say", json={"text": ""})

        assert response.status_code == 422

    def test_whitespace_only_text_returns_422(self):
        state = _make_state()
        app = _make_app(state)
        client = TestClient(app)

        response = client.post("/say", json={"text": "   "})

        assert response.status_code == 422

    def test_missing_text_returns_422(self):
        state = _make_state()
        app = _make_app(state)
        client = TestClient(app)

        response = client.post("/say", json={})

        assert response.status_code == 422

    def test_voice_override_accepted(self):
        state = _make_state(voices=["casual_female", "casual_male"])
        app = _make_app(state)
        client = TestClient(app)

        response = client.post("/say", json={"text": "Hello", "voice": "casual_male"})

        assert response.status_code == 202

    def test_unknown_voice_returns_400(self):
        state = _make_state(voices=["casual_female"])
        app = _make_app(state)
        client = TestClient(app)

        response = client.post("/say", json={"text": "Hello", "voice": "nonexistent"})

        assert response.status_code == 400
        assert "nonexistent" in response.json()["detail"]
        assert "casual_female" in response.json()["detail"]

    def test_creates_status_entry(self):
        state = _make_state()
        app = _make_app(state)
        client = TestClient(app)

        response = client.post("/say", json={"text": "Hello"})

        msg_id = response.json()["message_id"]
        with state.status_lock:
            assert msg_id in state.statuses
            ms = state.statuses[msg_id]
            assert ms.status == "queued"
            assert ms.text == "Hello"

    def test_queues_work_item(self):
        state = _make_state()
        app = _make_app(state)
        client = TestClient(app)

        client.post("/say", json={"text": "Hello", "voice": "casual_male"})

        item = state.work_queue.get_nowait()
        assert isinstance(item, WorkItem)
        assert item.text == "Hello"
        assert item.voice == "casual_male"

    def test_uses_default_voice_when_not_specified(self):
        state = _make_state(default_voice="casual_female")
        app = _make_app(state)
        client = TestClient(app)

        client.post("/say", json={"text": "Hello"})

        item = state.work_queue.get_nowait()
        assert item is not None
        assert item.voice == "casual_female"

    def test_applies_text_cleaning(self):
        state = _make_state()
        app = _make_app(state)
        client = TestClient(app)

        client.post("/say", json={"text": "  hello   world  "})

        item = state.work_queue.get_nowait()
        assert item is not None
        assert item.text == "hello world"

    def test_applies_simplify_punctuation_when_enabled(self):
        state = _make_state(simplify_punctuation=True)
        app = _make_app(state)
        client = TestClient(app)

        client.post("/say", json={"text": "Hello, world!"})

        item = state.work_queue.get_nowait()
        assert item is not None
        assert item.text == "Hello world."

    def test_multiple_requests_get_sequential_positions(self):
        state = _make_state()
        app = _make_app(state)
        client = TestClient(app)

        r1 = client.post("/say", json={"text": "First"})
        r2 = client.post("/say", json={"text": "Second"})
        r3 = client.post("/say", json={"text": "Third"})

        p1 = r1.json()["queue_position"]
        p2 = r2.json()["queue_position"]
        p3 = r3.json()["queue_position"]
        assert p1 < p2 < p3


class TestStatus:
    """Tests for GET /status/{message_id}."""

    def test_known_queued_message(self):
        state = _make_state()
        with state.status_lock:
            state.statuses["msg_test_001"] = MessageStatus(
                message_id="msg_test_001",
                status="queued",
                text="Hello",
                audio_file=None,
                error=None,
                completed_at=None,
            )
        app = _make_app(state)
        client = TestClient(app)

        response = client.get("/status/msg_test_001")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert data["text"] == "Hello"
        assert data["audio_file"] is None

    def test_known_completed_message(self):
        state = _make_state()
        with state.status_lock:
            state.statuses["msg_test_002"] = MessageStatus(
                message_id="msg_test_002",
                status="completed",
                text="Done",
                audio_file="data/output/speech_20260331_120000.wav",
                error=None,
                completed_at=time.time(),
            )
        app = _make_app(state)
        client = TestClient(app)

        response = client.get("/status/msg_test_002")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["audio_file"] == "data/output/speech_20260331_120000.wav"

    def test_known_errored_message(self):
        state = _make_state()
        with state.status_lock:
            state.statuses["msg_test_003"] = MessageStatus(
                message_id="msg_test_003",
                status="error",
                text="Bad",
                audio_file=None,
                error="Model failed",
                completed_at=time.time(),
            )
        app = _make_app(state)
        client = TestClient(app)

        response = client.get("/status/msg_test_003")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert data["error"] == "Model failed"

    def test_unknown_message_returns_404(self):
        state = _make_state()
        app = _make_app(state)
        client = TestClient(app)

        response = client.get("/status/nonexistent")

        assert response.status_code == 404


class TestEviction:
    """Tests for status entry eviction."""

    def test_expired_completed_entry_evicted(self):
        state = _make_state()
        expired_time = time.time() - STATUS_TTL_SECONDS - 1
        with state.status_lock:
            state.statuses["msg_old"] = MessageStatus(
                message_id="msg_old",
                status="completed",
                text="Old",
                audio_file="out.wav",
                error=None,
                completed_at=expired_time,
            )

        state.evict_expired()

        with state.status_lock:
            assert "msg_old" not in state.statuses

    def test_expired_error_entry_evicted(self):
        state = _make_state()
        expired_time = time.time() - STATUS_TTL_SECONDS - 1
        with state.status_lock:
            state.statuses["msg_err"] = MessageStatus(
                message_id="msg_err",
                status="error",
                text="Err",
                audio_file=None,
                error="failed",
                completed_at=expired_time,
            )

        state.evict_expired()

        with state.status_lock:
            assert "msg_err" not in state.statuses

    def test_queued_entry_never_evicted(self):
        state = _make_state()
        with state.status_lock:
            state.statuses["msg_queued"] = MessageStatus(
                message_id="msg_queued",
                status="queued",
                text="Waiting",
                audio_file=None,
                error=None,
                completed_at=None,
            )

        state.evict_expired()

        with state.status_lock:
            assert "msg_queued" in state.statuses

    def test_recent_completed_not_evicted(self):
        state = _make_state()
        with state.status_lock:
            state.statuses["msg_recent"] = MessageStatus(
                message_id="msg_recent",
                status="completed",
                text="Recent",
                audio_file="out.wav",
                error=None,
                completed_at=time.time(),
            )

        state.evict_expired()

        with state.status_lock:
            assert "msg_recent" in state.statuses

    def test_eviction_triggered_by_status_endpoint(self):
        state = _make_state()
        expired_time = time.time() - STATUS_TTL_SECONDS - 1
        with state.status_lock:
            state.statuses["msg_old"] = MessageStatus(
                message_id="msg_old",
                status="completed",
                text="Old",
                audio_file="out.wav",
                error=None,
                completed_at=expired_time,
            )
        app = _make_app(state)
        client = TestClient(app)

        response = client.get("/status/msg_old")

        assert response.status_code == 404


class TestServerAudioWorker:
    """Tests for the server audio worker."""

    def test_processes_single_message(self) -> None:
        state = _make_state()
        mock_chunk = MagicMock()
        mock_chunk.audio = np.ones(100, dtype=np.float32)
        model = cast(Any, state.model)
        model.generate.return_value = [mock_chunk]

        msg_id = "msg_test_001"
        with state.status_lock:
            state.statuses[msg_id] = MessageStatus(
                message_id=msg_id,
                status="queued",
                text="Hello",
                audio_file=None,
                error=None,
                completed_at=None,
            )
        state.work_queue.put(WorkItem(message_id=msg_id, text="Hello", voice="casual_female"))
        state.work_queue.put(None)

        t = threading.Thread(target=server_audio_worker, args=(state,))
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        with state.status_lock:
            assert state.statuses[msg_id].status == "completed"
            assert state.statuses[msg_id].audio_file is not None

    def test_processes_multiple_messages_sequentially(self) -> None:
        state = _make_state()
        mock_chunk = MagicMock()
        mock_chunk.audio = np.ones(100, dtype=np.float32)
        model = cast(Any, state.model)
        model.generate.return_value = [mock_chunk]

        for i in range(3):
            msg_id = f"msg_test_{i:03d}"
            with state.status_lock:
                state.statuses[msg_id] = MessageStatus(
                    message_id=msg_id,
                    status="queued",
                    text=f"Message {i}",
                    audio_file=None,
                    error=None,
                    completed_at=None,
                )
            state.work_queue.put(WorkItem(message_id=msg_id, text=f"Message {i}", voice="casual_female"))
        state.work_queue.put(None)

        t = threading.Thread(target=server_audio_worker, args=(state,))
        t.start()
        t.join(timeout=10)

        assert not t.is_alive()
        with state.status_lock:
            for i in range(3):
                msg_id = f"msg_test_{i:03d}"
                assert state.statuses[msg_id].status == "completed"

    def test_handles_generation_error(self) -> None:
        state = _make_state()
        model = cast(Any, state.model)
        model.generate.side_effect = RuntimeError("Model crashed")

        msg_id = "msg_err_001"
        with state.status_lock:
            state.statuses[msg_id] = MessageStatus(
                message_id=msg_id,
                status="queued",
                text="Fail",
                audio_file=None,
                error=None,
                completed_at=None,
            )
        state.work_queue.put(WorkItem(message_id=msg_id, text="Fail", voice="casual_female"))
        state.work_queue.put(None)

        t = threading.Thread(target=server_audio_worker, args=(state,))
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        with state.status_lock:
            assert state.statuses[msg_id].status == "error"
            error = state.statuses[msg_id].error
            assert error is not None
            assert "Model crashed" in error

    def test_shuts_down_on_none_sentinel(self) -> None:
        state = _make_state()
        state.work_queue.put(None)

        t = threading.Thread(target=server_audio_worker, args=(state,))
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        model = cast(Any, state.model)
        model.generate.assert_not_called()

    def test_handles_playback_error(self) -> None:
        state = _make_state()
        mock_chunk = MagicMock()
        mock_chunk.audio = np.ones(100, dtype=np.float32)
        model = cast(Any, state.model)
        model.generate.return_value = [mock_chunk]
        _ImmediateAudioPlayer.playback_error = RuntimeError("Audio device error")

        msg_id = "msg_play_err"
        with state.status_lock:
            state.statuses[msg_id] = MessageStatus(
                message_id=msg_id,
                status="queued",
                text="Hello",
                audio_file=None,
                error=None,
                completed_at=None,
            )
        state.work_queue.put(WorkItem(message_id=msg_id, text="Hello", voice="casual_female"))
        state.work_queue.put(None)

        t = threading.Thread(target=server_audio_worker, args=(state,))
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        with state.status_lock:
            assert state.statuses[msg_id].status == "error"
            error = state.statuses[msg_id].error
            assert error is not None
            assert "Audio device error" in error

    @patch("src.tts.sd")
    def test_recovers_after_output_stream_terminates(
        self,
        mock_sd: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("src.server.AudioPlayer", AudioPlayer)

        first_stream = MagicMock()
        second_stream = MagicMock()
        first_stream.write.side_effect = [None, RuntimeError("output stream terminated")]
        mock_sd.OutputStream.side_effect = [first_stream, second_stream]

        state = _make_state(save_wav=False, sample_rate=1000, lead_silence_ms=200)
        mock_chunk = MagicMock()
        mock_chunk.audio = np.ones(4, dtype=np.float32)
        model = cast(Any, state.model)
        model.generate.return_value = [mock_chunk]

        first_msg = "msg_stream_lost"
        second_msg = "msg_stream_recovered"
        for message_id, text in [(first_msg, "first"), (second_msg, "second")]:
            with state.status_lock:
                state.statuses[message_id] = MessageStatus(
                    message_id=message_id,
                    status="queued",
                    text=text,
                    audio_file=None,
                    error=None,
                    completed_at=None,
                )
            state.work_queue.put(WorkItem(message_id=message_id, text=text, voice="casual_female"))
        state.work_queue.put(None)

        t = threading.Thread(target=server_audio_worker, args=(state,))
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        assert mock_sd.OutputStream.call_count == 2

        first_silence = first_stream.write.call_args_list[0].args[0]
        second_silence = second_stream.write.call_args_list[0].args[0]
        assert first_silence.shape == (200, 1)
        assert second_silence.shape == (200, 1)
        assert float(np.max(np.abs(first_silence))) == 0.0
        assert float(np.max(np.abs(second_silence))) == 0.0

        with state.status_lock:
            assert state.statuses[first_msg].status == "error"
            assert state.statuses[first_msg].error is not None
            assert "output stream terminated" in state.statuses[first_msg].error
            assert state.statuses[second_msg].status == "completed"

    @patch("src.server.load")
    def test_loads_model_on_worker_thread_when_not_preloaded(self, mock_load: MagicMock) -> None:
        """Regression: the model must be loaded on the same thread that calls
        generate, because MLX GPU streams are thread-local. Loading on the main
        thread and generating on the worker thread raised
        'no Stream(gpu, 0) in current thread'.
        """
        load_thread_id: dict[str, int] = {}
        generate_thread_id: dict[str, int] = {}

        mock_chunk = MagicMock()
        mock_chunk.audio = np.ones(100, dtype=np.float32)

        mock_model = MagicMock()

        def fake_generate(text: str, voice: str) -> list[Any]:
            generate_thread_id["id"] = threading.get_ident()
            return [mock_chunk]

        mock_model.generate.side_effect = fake_generate

        def fake_load(model_path: str) -> MagicMock:
            load_thread_id["id"] = threading.get_ident()
            return mock_model

        mock_load.side_effect = fake_load

        state = _make_state(preload_model=False)

        msg_id = "msg_thread_001"
        with state.status_lock:
            state.statuses[msg_id] = MessageStatus(
                message_id=msg_id,
                status="queued",
                text="Hello",
                audio_file=None,
                error=None,
                completed_at=None,
            )
        state.work_queue.put(WorkItem(message_id=msg_id, text="Hello", voice="casual_female"))
        state.work_queue.put(None)

        t = threading.Thread(target=server_audio_worker, args=(state,))
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        mock_load.assert_called_once_with("test-model-path")
        assert load_thread_id["id"] == t.ident
        assert generate_thread_id["id"] == t.ident
        assert load_thread_id["id"] == generate_thread_id["id"]
        with state.status_lock:
            assert state.statuses[msg_id].status == "completed"

    @patch("src.server.load")
    def test_reports_model_load_failure_via_ready_queue(self, mock_load: MagicMock) -> None:
        """A model-load failure on the worker thread is propagated through
        ready_queue so the main thread (lifespan) can surface it on startup.
        """
        mock_load.side_effect = RuntimeError("model load boom")

        state = _make_state(preload_model=False)
        state.work_queue.put(None)

        t = threading.Thread(target=server_audio_worker, args=(state,))
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        result = state.ready_queue.get(timeout=1)
        assert isinstance(result, RuntimeError)
        assert "model load boom" in str(result)

    def test_signals_ready_when_model_preloaded(self) -> None:
        """When a model is pre-loaded, the worker still signals readiness with
        None so the lifespan startup unblocks.
        """
        state = _make_state()
        state.work_queue.put(None)

        t = threading.Thread(target=server_audio_worker, args=(state,))
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        assert state.ready_queue.get(timeout=1) is None


class TestSaveWavDisabled:
    """Tests for save_wav=False behavior."""

    def test_completes_without_audio_file_when_save_wav_disabled(self) -> None:
        state = _make_state(save_wav=False)
        mock_chunk = MagicMock()
        mock_chunk.audio = np.ones(100, dtype=np.float32)
        model = cast(Any, state.model)
        model.generate.return_value = [mock_chunk]

        msg_id = "msg_nosave_001"
        with state.status_lock:
            state.statuses[msg_id] = MessageStatus(
                message_id=msg_id,
                status="queued",
                text="Hello",
                audio_file=None,
                error=None,
                completed_at=None,
            )
        state.work_queue.put(WorkItem(message_id=msg_id, text="Hello", voice="casual_female"))
        state.work_queue.put(None)

        t = threading.Thread(target=server_audio_worker, args=(state,))
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        with state.status_lock:
            assert state.statuses[msg_id].status == "completed"
            assert state.statuses[msg_id].audio_file is None


class TestConcurrentSay:
    """Test concurrent /say requests are all accepted and processed sequentially."""

    def test_concurrent_requests_all_complete(self) -> None:
        state = _make_state()
        mock_chunk = MagicMock()
        mock_chunk.audio = np.ones(100, dtype=np.float32)
        model = cast(Any, state.model)
        model.generate.return_value = [mock_chunk]

        app = _make_app(state)
        client = TestClient(app)

        num_messages = 5
        responses: list[dict[str, Any]] = [{}] * num_messages

        def send_say(index: int) -> None:
            resp = client.post("/say", json={"text": f"Message {index}"})
            responses[index] = resp.json()

        threads = [threading.Thread(target=send_say, args=(i,)) for i in range(num_messages)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # All accepted with 'queued' status
        for i, r in enumerate(responses):
            assert "message_id" in r, f"Message {i} was not accepted"
            assert r["status"] == "queued"

        # Start worker to process the queue
        state.work_queue.put(None)
        worker = threading.Thread(target=server_audio_worker, args=(state,))
        worker.start()
        worker.join(timeout=15)

        assert not worker.is_alive()

        # All messages completed
        with state.status_lock:
            for r in responses:
                ms = state.statuses[r["message_id"]]
                assert ms.status == "completed"

        # Worker called generate once per message
        assert model.generate.call_count == num_messages

    def test_concurrent_requests_no_overlapping_playback(self) -> None:
        state = _make_state()
        mock_chunk = MagicMock()
        mock_chunk.audio = np.ones(100, dtype=np.float32)
        model = cast(Any, state.model)
        model.generate.return_value = [mock_chunk]

        app = _make_app(state)
        client = TestClient(app)

        num_messages = 5
        for i in range(num_messages):
            client.post("/say", json={"text": f"Overlap test {i}"})

        state.work_queue.put(None)
        worker = threading.Thread(target=server_audio_worker, args=(state,))
        worker.start()
        worker.join(timeout=15)

        assert not worker.is_alive()
        assert _ImmediateAudioPlayer.max_active_count == 1


class TestTextPipelineIntegration:
    """Integration test: clean_text + simplify_punctuation compose correctly."""

    def test_cleaning_then_simplification(self):
        state = _make_state(simplify_punctuation=True)
        app = _make_app(state)
        client = TestClient(app)

        client.post("/say", json={"text": "  Hello,   world!  How  are  you?  "})

        item = state.work_queue.get_nowait()
        assert item is not None
        assert item.text == "Hello world. How are you."
