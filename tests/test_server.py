"""Tests for the FastAPI TTS server."""

import re
import threading
import time
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
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


def _make_state(
    voices: list[str] | None = None,
    default_voice: str = "casual_female",
    sample_rate: int = 24000,
    simplify_punctuation: bool = False,
) -> ServerState:
    """Create a ServerState with a mock model for testing."""
    model = MagicMock()
    if voices is None:
        voices = ["casual_female", "casual_male"]
    return ServerState(
        model=model,
        voices=voices,
        default_voice=default_voice,
        sample_rate=sample_rate,
        simplify_punctuation=simplify_punctuation,
    )


def _make_app(state: ServerState) -> FastAPI:
    """Create a test FastAPI app with the given state."""
    app = FastAPI()
    app.state.server = state
    app.include_router(router)
    return app


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

    @patch("src.server.play_chunks")
    def test_processes_single_message(self, mock_play: MagicMock) -> None:
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

    @patch("src.server.play_chunks")
    def test_processes_multiple_messages_sequentially(self, mock_play: MagicMock) -> None:
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

    @patch("src.server.play_chunks")
    def test_handles_generation_error(self, mock_play: MagicMock) -> None:
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

    @patch("src.server.play_chunks")
    def test_shuts_down_on_none_sentinel(self, mock_play: MagicMock) -> None:
        state = _make_state()
        state.work_queue.put(None)

        t = threading.Thread(target=server_audio_worker, args=(state,))
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        model = cast(Any, state.model)
        model.generate.assert_not_called()

    @patch("src.server.play_chunks")
    def test_handles_playback_error(self, mock_play: MagicMock) -> None:
        state = _make_state()
        mock_chunk = MagicMock()
        mock_chunk.audio = np.ones(100, dtype=np.float32)
        model = cast(Any, state.model)
        model.generate.return_value = [mock_chunk]
        mock_play.side_effect = RuntimeError("Audio device error")

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


class TestConcurrentSay:
    """Test concurrent /say requests are all accepted and processed sequentially."""

    @patch("src.server.play_chunks")
    def test_concurrent_requests_all_complete(self, mock_play: MagicMock) -> None:
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

    @patch("src.server.play_chunks")
    def test_concurrent_requests_no_overlapping_playback(self, mock_play: MagicMock) -> None:
        playing_count = {"current": 0, "max": 0}
        lock = threading.Lock()

        original_side_effect = mock_play.side_effect

        def track_playback(*args: Any, **kwargs: Any) -> None:
            with lock:
                playing_count["current"] += 1
                if playing_count["current"] > playing_count["max"]:
                    playing_count["max"] = playing_count["current"]
            time.sleep(0.05)  # simulate playback duration
            with lock:
                playing_count["current"] -= 1
            if original_side_effect:
                return original_side_effect(*args, **kwargs)
            return None

        mock_play.side_effect = track_playback

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
        assert playing_count["max"] == 1, f"Max concurrent playback was {playing_count['max']}, expected 1"


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
