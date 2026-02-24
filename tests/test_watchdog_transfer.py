"""Tests for the watchdog file transfer logic."""

from unittest.mock import AsyncMock

import pytest

from corvus.schemas.watchdog import TransferMethod, TransferStatus
from corvus.watchdog.audit import WatchdogAuditLog
from corvus.watchdog.hash_store import HashStore
from corvus.watchdog.transfer import compute_file_hash, process_file, transfer_by_move

# ------------------------------------------------------------------
# compute_file_hash
# ------------------------------------------------------------------


class TestComputeFileHash:
    def test_hash_known_content(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")
        h = compute_file_hash(f)
        # SHA-256 of "hello world"
        assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    def test_hash_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        h = compute_file_hash(f)
        # SHA-256 of empty string
        assert h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_same_content_same_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"same content")
        f2.write_bytes(b"same content")
        assert compute_file_hash(f1) == compute_file_hash(f2)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert compute_file_hash(f1) != compute_file_hash(f2)


# ------------------------------------------------------------------
# transfer_by_move
# ------------------------------------------------------------------


class TestTransferByMove:
    def test_moves_file(self, tmp_path):
        src = tmp_path / "scans" / "doc.pdf"
        dest_dir = tmp_path / "consume"
        src.parent.mkdir()
        dest_dir.mkdir()
        src.write_bytes(b"PDF content")

        result = transfer_by_move(src, dest_dir)

        assert result == dest_dir / "doc.pdf"
        assert result.exists()
        assert result.read_bytes() == b"PDF content"
        assert not src.exists()

    def test_name_collision_appends_suffix(self, tmp_path):
        src = tmp_path / "scans" / "doc.pdf"
        dest_dir = tmp_path / "consume"
        src.parent.mkdir()
        dest_dir.mkdir()
        src.write_bytes(b"new content")
        # Pre-existing file at destination
        (dest_dir / "doc.pdf").write_bytes(b"existing")

        result = transfer_by_move(src, dest_dir)

        assert result == dest_dir / "doc_1.pdf"
        assert result.read_bytes() == b"new content"
        assert (dest_dir / "doc.pdf").read_bytes() == b"existing"

    def test_multiple_collisions(self, tmp_path):
        src = tmp_path / "scans" / "doc.pdf"
        dest_dir = tmp_path / "consume"
        src.parent.mkdir()
        dest_dir.mkdir()
        src.write_bytes(b"newest")
        (dest_dir / "doc.pdf").write_bytes(b"v0")
        (dest_dir / "doc_1.pdf").write_bytes(b"v1")

        result = transfer_by_move(src, dest_dir)

        assert result == dest_dir / "doc_2.pdf"


# ------------------------------------------------------------------
# process_file
# ------------------------------------------------------------------


@pytest.fixture
def processing_env(tmp_path):
    """Set up a complete processing environment."""
    scan_dir = tmp_path / "scans"
    consume_dir = tmp_path / "consume"
    scan_dir.mkdir()
    consume_dir.mkdir()

    test_file = scan_dir / "invoice.pdf"
    test_file.write_bytes(b"fake PDF content")

    return {
        "scan_dir": scan_dir,
        "consume_dir": consume_dir,
        "test_file": test_file,
        "hash_store": HashStore(tmp_path / "hashes.db"),
        "audit_log": WatchdogAuditLog(tmp_path / "audit.jsonl"),
    }


class TestProcessFileMoveMethod:
    async def test_success(self, processing_env):
        env = processing_env
        event = await process_file(
            env["test_file"],
            method=TransferMethod.MOVE,
            hash_store=env["hash_store"],
            audit_log=env["audit_log"],
            consume_dir=env["consume_dir"],
        )

        assert event.transfer_status == TransferStatus.SUCCESS
        assert event.file_name == "invoice.pdf"
        assert str(env["consume_dir"]) in event.destination
        assert env["hash_store"].count() == 1
        assert len(env["audit_log"].read_entries()) == 1

    async def test_duplicate_detected(self, processing_env):
        env = processing_env
        # First run
        await process_file(
            env["test_file"],
            method=TransferMethod.MOVE,
            hash_store=env["hash_store"],
            audit_log=env["audit_log"],
            consume_dir=env["consume_dir"],
        )

        # Create another file with the same content
        dup_file = env["scan_dir"] / "invoice_copy.pdf"
        dup_file.write_bytes(b"fake PDF content")

        event = await process_file(
            dup_file,
            method=TransferMethod.MOVE,
            hash_store=env["hash_store"],
            audit_log=env["audit_log"],
            consume_dir=env["consume_dir"],
        )

        assert event.transfer_status == TransferStatus.DUPLICATE
        assert env["hash_store"].count() == 1  # No new hash added
        assert len(env["audit_log"].read_entries()) == 2  # Both events logged

    async def test_missing_consume_dir_raises(self, processing_env):
        env = processing_env
        event = await process_file(
            env["test_file"],
            method=TransferMethod.MOVE,
            hash_store=env["hash_store"],
            audit_log=env["audit_log"],
            consume_dir=None,
        )
        assert event.transfer_status == TransferStatus.ERROR
        assert "consume_dir" in event.error_message


class TestProcessFileUploadMethod:
    async def test_success(self, processing_env):
        env = processing_env
        mock_client = AsyncMock()
        mock_client.upload_document.return_value = "task-uuid-abc123"

        event = await process_file(
            env["test_file"],
            method=TransferMethod.UPLOAD,
            hash_store=env["hash_store"],
            audit_log=env["audit_log"],
            paperless_client=mock_client,
        )

        assert event.transfer_status == TransferStatus.SUCCESS
        assert event.destination == "task-uuid-abc123"
        mock_client.upload_document.assert_awaited_once_with(env["test_file"])

    async def test_upload_error(self, processing_env):
        env = processing_env
        mock_client = AsyncMock()
        mock_client.upload_document.side_effect = Exception("Connection refused")

        event = await process_file(
            env["test_file"],
            method=TransferMethod.UPLOAD,
            hash_store=env["hash_store"],
            audit_log=env["audit_log"],
            paperless_client=mock_client,
        )

        assert event.transfer_status == TransferStatus.ERROR
        assert "Connection refused" in event.error_message
        assert env["hash_store"].count() == 0  # Not recorded on error

    async def test_missing_client_raises(self, processing_env):
        env = processing_env
        event = await process_file(
            env["test_file"],
            method=TransferMethod.UPLOAD,
            hash_store=env["hash_store"],
            audit_log=env["audit_log"],
            paperless_client=None,
        )
        assert event.transfer_status == TransferStatus.ERROR
        assert "paperless_client" in event.error_message


class TestProcessFileMetadata:
    async def test_event_has_correct_metadata(self, processing_env):
        env = processing_env
        event = await process_file(
            env["test_file"],
            method=TransferMethod.MOVE,
            hash_store=env["hash_store"],
            audit_log=env["audit_log"],
            consume_dir=env["consume_dir"],
        )

        assert event.file_name == "invoice.pdf"
        assert event.source_path == str(env["test_file"])
        assert len(event.file_hash) == 64  # SHA-256 hex digest
        assert event.file_size_bytes == len(b"fake PDF content")
        assert event.transfer_method == TransferMethod.MOVE
