"""Tests for the Corvus web interface (S21.1 + S21.2)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def api_key():
    return "test-key-12345"


@pytest.fixture()
def client(api_key):
    """TestClient with a known API key configured."""
    with patch("corvus.web.auth.API_KEY", api_key):
        from corvus.web.app import app

        yield TestClient(app)


@pytest.fixture()
def client_no_key():
    """TestClient with no API key configured (empty string)."""
    with patch("corvus.web.auth.API_KEY", ""):
        from corvus.web.app import app

        yield TestClient(app)


def _authed(api_key):
    return {"X-API-Key": api_key}


def _make_doc_task(*, doc_id=1, confidence=0.9, gate="auto_execute"):
    from corvus.schemas.document_tagging import (
        DocumentTaggingResult,
        DocumentTaggingTask,
        GateAction,
        TagSuggestion,
    )

    return DocumentTaggingTask(
        document_id=doc_id,
        document_title=f"Doc {doc_id}",
        content_snippet="Test content...",
        result=DocumentTaggingResult(
            suggested_tags=[TagSuggestion(tag_name="test", confidence=confidence)],
            reasoning="test",
        ),
        overall_confidence=confidence,
        gate_action=GateAction(gate),
    )


def _make_doc_update(*, doc_id=1):
    from corvus.schemas.document_tagging import ProposedDocumentUpdate

    return ProposedDocumentUpdate(document_id=doc_id)


def _make_email_task(*, uid="1", confidence=0.8, gate="auto_execute"):
    from corvus.schemas.document_tagging import GateAction
    from corvus.schemas.email import (
        EmailAction,
        EmailActionType,
        EmailCategory,
        EmailClassification,
        EmailTriageTask,
    )

    return EmailTriageTask(
        uid=uid,
        account_email="test@test.com",
        subject=f"Email {uid}",
        from_address="sender@test.com",
        classification=EmailClassification(
            category=EmailCategory.PERSONAL,
            confidence=confidence,
            reasoning="test",
            suggested_action="keep",
        ),
        proposed_action=EmailAction(action_type=EmailActionType.KEEP),
        overall_confidence=confidence,
        gate_action=GateAction(gate),
    )


# ── Health ───────────────────────────────────────────────────────────


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_no_auth_needed(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200


# ── Auth ─────────────────────────────────────────────────────────────


class TestAuth:
    def test_valid_key(self, client, api_key):
        resp = client.get("/api/status", headers=_authed(api_key))
        assert resp.status_code == 200

    def test_missing_key(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 401

    def test_wrong_key(self, client):
        resp = client.get("/api/status", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_no_key_configured(self, client_no_key):
        resp = client_no_key.get("/api/status", headers={"X-API-Key": "anything"})
        assert resp.status_code == 500
        assert "not configured" in resp.json()["detail"]


# ── Status ───────────────────────────────────────────────────────────


class TestStatus:
    def _patch_paths(self, tmp_path):
        return (
            patch("corvus.config.QUEUE_DB_PATH", str(tmp_path / "queue.db")),
            patch("corvus.config.AUDIT_LOG_PATH", str(tmp_path / "audit.log")),
            patch("corvus.config.EMAIL_REVIEW_DB_PATH", str(tmp_path / "email_queue.db")),
            patch("corvus.config.EMAIL_AUDIT_LOG_PATH", str(tmp_path / "email_audit.log")),
        )

    def test_status_returns_both_sections(self, client, api_key, tmp_path):
        with self._patch_paths(tmp_path)[0], self._patch_paths(tmp_path)[1], \
             self._patch_paths(tmp_path)[2], self._patch_paths(tmp_path)[3]:
            resp = client.get("/api/status", headers=_authed(api_key))

        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data
        assert "email" in data
        assert data["documents"]["pending_review"] == 0
        assert data["email"]["pending_review"] == 0

    def test_status_reflects_pending_counts(self, client, api_key, tmp_path):
        from corvus.queue.email_review import EmailReviewQueue
        from corvus.queue.review import ReviewQueue

        db_path = str(tmp_path / "queue.db")
        email_db_path = str(tmp_path / "email_queue.db")

        with ReviewQueue(db_path) as q:
            q.add(_make_doc_task(gate="queue_for_review"), _make_doc_update())

        with EmailReviewQueue(email_db_path) as q:
            q.add(_make_email_task(gate="queue_for_review"))

        with (
            patch("corvus.config.QUEUE_DB_PATH", db_path),
            patch("corvus.config.AUDIT_LOG_PATH", str(tmp_path / "audit.log")),
            patch("corvus.config.EMAIL_REVIEW_DB_PATH", email_db_path),
            patch("corvus.config.EMAIL_AUDIT_LOG_PATH", str(tmp_path / "email_audit.log")),
        ):
            resp = client.get("/api/status", headers=_authed(api_key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["documents"]["pending_review"] == 1
        assert data["email"]["pending_review"] == 1


# ── Audit: Documents ────────────────────────────────────────────────


class TestAuditDocuments:
    def test_empty_log(self, client, api_key, tmp_path):
        with patch("corvus.config.AUDIT_LOG_PATH", str(tmp_path / "audit.log")):
            resp = client.get("/api/audit/documents", headers=_authed(api_key))
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_entries(self, client, api_key, tmp_path):
        from corvus.audit.logger import AuditLog

        audit_path = str(tmp_path / "audit.log")
        audit = AuditLog(audit_path)
        task = _make_doc_task(doc_id=42)
        audit.log_auto_applied(task, _make_doc_update(doc_id=42))

        with patch("corvus.config.AUDIT_LOG_PATH", audit_path):
            resp = client.get("/api/audit/documents", headers=_authed(api_key))

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["document_id"] == 42
        assert data[0]["action"] == "auto_applied"

    def test_limit_param(self, client, api_key, tmp_path):
        from corvus.audit.logger import AuditLog

        audit_path = str(tmp_path / "audit.log")
        audit = AuditLog(audit_path)
        task = _make_doc_task()
        update = _make_doc_update()
        for _ in range(5):
            audit.log_auto_applied(task, update)

        with patch("corvus.config.AUDIT_LOG_PATH", audit_path):
            resp = client.get("/api/audit/documents?limit=2", headers=_authed(api_key))

        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_since_param(self, client, api_key, tmp_path):
        from corvus.audit.logger import AuditLog

        audit_path = str(tmp_path / "audit.log")
        audit = AuditLog(audit_path)
        audit.log_auto_applied(_make_doc_task(), _make_doc_update())

        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        with patch("corvus.config.AUDIT_LOG_PATH", audit_path):
            resp = client.get(
                "/api/audit/documents", params={"since": future},
                headers=_authed(api_key),
            )

        assert resp.status_code == 200
        assert resp.json() == []


# ── Audit: Email ─────────────────────────────────────────────────────


class TestAuditEmail:
    def test_empty_log(self, client, api_key, tmp_path):
        with patch("corvus.config.EMAIL_AUDIT_LOG_PATH", str(tmp_path / "email_audit.log")):
            resp = client.get("/api/audit/email", headers=_authed(api_key))
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_entries(self, client, api_key, tmp_path):
        from corvus.audit.email_logger import EmailAuditLog

        audit_path = str(tmp_path / "email_audit.log")
        audit = EmailAuditLog(audit_path)
        audit.log_auto_applied(_make_email_task(uid="123"))

        with patch("corvus.config.EMAIL_AUDIT_LOG_PATH", audit_path):
            resp = client.get("/api/audit/email", headers=_authed(api_key))

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["subject"] == "Email 123"
        assert data[0]["action"] == "auto_applied"


# ── Audit: Watchdog ──────────────────────────────────────────────────


class TestAuditWatchdog:
    def test_empty_log(self, client, api_key, tmp_path):
        with patch("corvus.config.WATCHDOG_AUDIT_LOG_PATH", str(tmp_path / "wd.log")):
            resp = client.get("/api/audit/watchdog", headers=_authed(api_key))
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_entries(self, client, api_key, tmp_path):
        from corvus.schemas.watchdog import TransferMethod, TransferStatus, WatchdogEvent
        from corvus.watchdog.audit import WatchdogAuditLog

        audit_path = str(tmp_path / "wd.log")
        audit = WatchdogAuditLog(audit_path)
        audit.log(
            WatchdogEvent(
                timestamp=datetime.now(UTC),
                source_path="/scan/test.pdf",
                file_name="test.pdf",
                file_hash="abc123",
                transfer_method=TransferMethod.UPLOAD,
                transfer_status=TransferStatus.SUCCESS,
                destination="task-uuid-123",
                file_size_bytes=1024,
            )
        )

        with patch("corvus.config.WATCHDOG_AUDIT_LOG_PATH", audit_path):
            resp = client.get("/api/audit/watchdog", headers=_authed(api_key))

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["file_name"] == "test.pdf"
        assert data[0]["transfer_status"] == "success"


# ── Review: Documents ────────────────────────────────────────────────


class TestReviewDocuments:
    def _patch_paths(self, tmp_path):
        return (
            patch("corvus.config.QUEUE_DB_PATH", str(tmp_path / "queue.db")),
            patch("corvus.config.AUDIT_LOG_PATH", str(tmp_path / "audit.log")),
        )

    def _add_doc_item(self, db_path):
        from corvus.queue.review import ReviewQueue

        task = _make_doc_task(gate="queue_for_review")
        update = _make_doc_update()
        with ReviewQueue(db_path) as q:
            item = q.add(task, update)
        return item

    def test_list_empty(self, client, api_key, tmp_path):
        with self._patch_paths(tmp_path)[0], self._patch_paths(tmp_path)[1]:
            resp = client.get("/api/review/documents", headers=_authed(api_key))
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_pending(self, client, api_key, tmp_path):
        db_path = str(tmp_path / "queue.db")
        self._add_doc_item(db_path)

        with patch("corvus.config.QUEUE_DB_PATH", db_path):
            resp = client.get("/api/review/documents", headers=_authed(api_key))

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["status"] == "pending"

    def test_reject(self, client, api_key, tmp_path):
        db_path = str(tmp_path / "queue.db")
        audit_path = str(tmp_path / "audit.log")
        item = self._add_doc_item(db_path)

        with (
            patch("corvus.config.QUEUE_DB_PATH", db_path),
            patch("corvus.config.AUDIT_LOG_PATH", audit_path),
        ):
            resp = client.post(
                f"/api/review/documents/{item.id}/reject",
                json={"notes": "Not correct"},
                headers=_authed(api_key),
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"

        # Verify it's no longer pending
        from corvus.queue.review import ReviewQueue

        with ReviewQueue(db_path) as q:
            assert q.count_pending() == 0

    def test_reject_not_found(self, client, api_key, tmp_path):
        with self._patch_paths(tmp_path)[0], self._patch_paths(tmp_path)[1]:
            resp = client.post(
                "/api/review/documents/nonexistent/reject",
                headers=_authed(api_key),
            )
        assert resp.status_code == 404

    def test_reject_already_reviewed(self, client, api_key, tmp_path):
        db_path = str(tmp_path / "queue.db")
        audit_path = str(tmp_path / "audit.log")
        item = self._add_doc_item(db_path)

        # Reject it first
        from corvus.queue.review import ReviewQueue

        with ReviewQueue(db_path) as q:
            q.reject(item.id, notes="first rejection")

        with (
            patch("corvus.config.QUEUE_DB_PATH", db_path),
            patch("corvus.config.AUDIT_LOG_PATH", audit_path),
        ):
            resp = client.post(
                f"/api/review/documents/{item.id}/reject",
                headers=_authed(api_key),
            )
        assert resp.status_code == 409


# ── Review: Email ────────────────────────────────────────────────────


class TestReviewEmail:
    def _add_email_item(self, db_path):
        from corvus.queue.email_review import EmailReviewQueue

        task = _make_email_task(gate="queue_for_review")
        with EmailReviewQueue(db_path) as q:
            item = q.add(task)
        return item

    def test_list_empty(self, client, api_key, tmp_path):
        with patch("corvus.config.EMAIL_REVIEW_DB_PATH", str(tmp_path / "eq.db")):
            resp = client.get("/api/review/email", headers=_authed(api_key))
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_pending(self, client, api_key, tmp_path):
        db_path = str(tmp_path / "eq.db")
        self._add_email_item(db_path)

        with patch("corvus.config.EMAIL_REVIEW_DB_PATH", db_path):
            resp = client.get("/api/review/email", headers=_authed(api_key))

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["status"] == "pending"
        assert data[0]["task"]["subject"] == "Email 1"

    def test_reject(self, client, api_key, tmp_path):
        db_path = str(tmp_path / "eq.db")
        audit_path = str(tmp_path / "email_audit.log")
        item = self._add_email_item(db_path)

        with (
            patch("corvus.config.EMAIL_REVIEW_DB_PATH", db_path),
            patch("corvus.config.EMAIL_AUDIT_LOG_PATH", audit_path),
        ):
            resp = client.post(
                f"/api/review/email/{item.id}/reject",
                json={"notes": "Not spam"},
                headers=_authed(api_key),
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"

        from corvus.queue.email_review import EmailReviewQueue

        with EmailReviewQueue(db_path) as q:
            assert q.count_pending() == 0

    def test_reject_not_found(self, client, api_key, tmp_path):
        with patch("corvus.config.EMAIL_REVIEW_DB_PATH", str(tmp_path / "eq.db")):
            resp = client.post(
                "/api/review/email/nonexistent/reject",
                headers=_authed(api_key),
            )
        assert resp.status_code == 404

    def test_reject_already_reviewed(self, client, api_key, tmp_path):
        db_path = str(tmp_path / "eq.db")
        audit_path = str(tmp_path / "email_audit.log")
        item = self._add_email_item(db_path)

        from corvus.queue.email_review import EmailReviewQueue

        with EmailReviewQueue(db_path) as q:
            q.reject(item.id, notes="first")

        with (
            patch("corvus.config.EMAIL_REVIEW_DB_PATH", db_path),
            patch("corvus.config.EMAIL_AUDIT_LOG_PATH", audit_path),
        ):
            resp = client.post(
                f"/api/review/email/{item.id}/reject",
                headers=_authed(api_key),
            )
        assert resp.status_code == 409

    def test_approve_no_account(self, client, api_key, tmp_path):
        """Approve fails gracefully if no matching account config."""
        db_path = str(tmp_path / "eq.db")
        audit_path = str(tmp_path / "email_audit.log")
        item = self._add_email_item(db_path)

        with (
            patch("corvus.config.EMAIL_REVIEW_DB_PATH", db_path),
            patch("corvus.config.EMAIL_AUDIT_LOG_PATH", audit_path),
            patch("corvus.config.load_email_accounts", return_value=[]),
        ):
            resp = client.post(
                f"/api/review/email/{item.id}/approve",
                headers=_authed(api_key),
            )
        assert resp.status_code == 400
        assert "No account config" in resp.json()["detail"]


# ── CLI ──────────────────────────────────────────────────────────────


class TestServeCLI:
    def test_serve_command_exists(self):
        from click.testing import CliRunner

        from corvus.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Start the Corvus web server" in result.output

    def test_serve_help_shows_options(self):
        from click.testing import CliRunner

        from corvus.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--reload" in result.output


# ── Voice WebSocket ─────────────────────────────────────────────────


class TestVoiceWebSocket:
    """Tests for the /ws/voice WebSocket endpoint."""

    def test_ws_rejects_missing_api_key(self, client):
        """WS with no api_key query param gets closed with 4001."""
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/voice"):
                pass
        assert exc_info.value.code == 4001

    def test_ws_rejects_wrong_api_key(self, client):
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/voice?api_key=wrong"):
                pass
        assert exc_info.value.code == 4001

    def test_ws_accepts_valid_api_key(self, client, api_key):
        """Valid key gets a 'ready' message."""
        with patch("corvus.web.voice_ws.API_KEY", api_key):
            with client.websocket_connect(f"/ws/voice?api_key={api_key}") as ws:
                msg = ws.receive_json()
                assert msg["type"] == "ready"

    def test_ws_ping_pong(self, client, api_key):
        with patch("corvus.web.voice_ws.API_KEY", api_key):
            with client.websocket_connect(f"/ws/voice?api_key={api_key}") as ws:
                ws.receive_json()  # ready
                ws.send_json({"type": "ping"})
                msg = ws.receive_json()
                assert msg["type"] == "pong"

    def test_ws_audio_round_trip(self, client, api_key):
        """Send audio_start + audio data + audio_end, expect transcription."""
        import numpy as np

        mock_result = type("R", (), {"text": "hello world", "language": "en",
                                     "duration_seconds": 1.0, "no_speech_probability": 0.1})()

        mock_stt = type("STT", (), {"transcribe": lambda self, x: mock_result})()
        # Make transcribe async
        import asyncio
        async def mock_transcribe(audio):
            return mock_result
        mock_stt.transcribe = mock_transcribe

        mock_response = None

        async def mock_process_text(text, history):
            from corvus.schemas.orchestrator import (
                Intent,
                OrchestratorAction,
                OrchestratorResponse,
            )
            return OrchestratorResponse(
                action=OrchestratorAction.CHAT_RESPONSE,
                intent=Intent.GENERAL_CHAT,
                confidence=0.95,
                message="Hello there!",
            )

        mock_tts_audio = np.zeros(2400, dtype=np.float32)

        async def mock_synth_full(self, text):
            return mock_tts_audio, 24000

        mock_tts = type("TTS", (), {"synthesize_full": mock_synth_full})()

        async def mock_get_stt():
            return mock_stt

        async def mock_get_tts():
            return mock_tts

        with (
            patch("corvus.web.voice_ws.API_KEY", api_key),
            patch("corvus.web.voice_ws.voice_models.get_stt", mock_get_stt),
            patch("corvus.web.voice_ws.voice_models.get_tts", mock_get_tts),
            patch("corvus.web.voice_ws._process_text", mock_process_text),
            patch("corvus.orchestrator.conversation_store.ConversationStore") as MockStore,
        ):
            mock_store_inst = MockStore.return_value
            mock_store_inst.create.return_value = "conv-123"

            with client.websocket_connect(f"/ws/voice?api_key={api_key}") as ws:
                ready = ws.receive_json()
                assert ready["type"] == "ready"

                # Send audio
                ws.send_json({"type": "audio_start"})
                listening = ws.receive_json()
                assert listening["type"] == "listening"

                # Send some PCM audio bytes
                audio = np.random.randn(16000).astype(np.float32)
                ws.send_bytes(audio.tobytes())
                ws.send_json({"type": "audio_end"})

                # Should get: transcription, thinking, response, audio_start, audio bytes, audio_end
                transcription = ws.receive_json()
                assert transcription["type"] == "transcription"
                assert transcription["text"] == "hello world"

                thinking = ws.receive_json()
                assert thinking["type"] == "thinking"

                response = ws.receive_json()
                assert response["type"] == "response"
                assert "Hello" in response["text"]

                audio_start = ws.receive_json()
                assert audio_start["type"] == "audio_start"
                assert audio_start["sample_rate"] == 24000

                # Receive binary audio chunk(s)
                audio_data = ws.receive()
                assert "bytes" in audio_data

                audio_end = ws.receive_json()
                assert audio_end["type"] == "audio_end"

    def test_ws_empty_audio(self, client, api_key):
        """Empty audio (immediate audio_end) returns empty transcription."""
        mock_stt = type("STT", (), {})()

        async def mock_get_stt():
            return mock_stt

        with (
            patch("corvus.web.voice_ws.API_KEY", api_key),
            patch("corvus.web.voice_ws.voice_models.get_stt", mock_get_stt),
            patch("corvus.orchestrator.conversation_store.ConversationStore"),
        ):
            with client.websocket_connect(f"/ws/voice?api_key={api_key}") as ws:
                ws.receive_json()  # ready

                ws.send_json({"type": "audio_start"})
                ws.receive_json()  # listening

                ws.send_json({"type": "audio_end"})

                transcription = ws.receive_json()
                assert transcription["type"] == "transcription"
                assert transcription["text"] == ""
