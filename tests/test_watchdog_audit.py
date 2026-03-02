"""Tests for the watchdog JSONL audit log."""

from datetime import UTC, datetime, timedelta

from corvus.schemas.watchdog import TransferMethod, TransferStatus, WatchdogEvent
from corvus.watchdog.audit import WatchdogAuditLog


def _make_event(
    *,
    file_name: str = "scan.pdf",
    status: TransferStatus = TransferStatus.SUCCESS,
    method: TransferMethod = TransferMethod.UPLOAD,
) -> WatchdogEvent:
    return WatchdogEvent(
        timestamp=datetime.now(UTC),
        source_path=f"/scans/{file_name}",
        file_name=file_name,
        file_hash="a" * 64,
        transfer_method=method,
        transfer_status=status,
        destination="task-uuid-123" if status == TransferStatus.SUCCESS else "",
        file_size_bytes=1024,
    )


class TestLogAndRead:
    def test_log_creates_file(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        audit = WatchdogAuditLog(log_path)
        audit.log(_make_event())
        assert log_path.exists()

    def test_roundtrip(self, tmp_path):
        audit = WatchdogAuditLog(tmp_path / "audit.jsonl")
        event = _make_event()
        audit.log(event)

        entries = audit.read_entries()
        assert len(entries) == 1
        assert entries[0].file_name == "scan.pdf"
        assert entries[0].transfer_status == TransferStatus.SUCCESS
        assert entries[0].file_hash == "a" * 64

    def test_multiple_entries(self, tmp_path):
        audit = WatchdogAuditLog(tmp_path / "audit.jsonl")
        audit.log(_make_event(file_name="a.pdf"))
        audit.log(_make_event(file_name="b.pdf"))
        audit.log(_make_event(file_name="c.pdf"))

        entries = audit.read_entries()
        assert len(entries) == 3
        assert [e.file_name for e in entries] == ["a.pdf", "b.pdf", "c.pdf"]

    def test_read_empty_returns_empty_list(self, tmp_path):
        audit = WatchdogAuditLog(tmp_path / "audit.jsonl")
        assert audit.read_entries() == []


class TestFiltering:
    def test_filter_by_since(self, tmp_path):
        audit = WatchdogAuditLog(tmp_path / "audit.jsonl")
        now = datetime.now(UTC)
        old_event = _make_event(file_name="old.pdf")
        old_event.timestamp = now - timedelta(seconds=10)
        audit.log(old_event)

        cutoff = now - timedelta(seconds=5)

        new_event = _make_event(file_name="new.pdf")
        new_event.timestamp = now
        audit.log(new_event)

        entries = audit.read_entries(since=cutoff)
        assert len(entries) == 1
        assert entries[0].file_name == "new.pdf"

    def test_filter_by_status(self, tmp_path):
        audit = WatchdogAuditLog(tmp_path / "audit.jsonl")
        audit.log(_make_event(status=TransferStatus.SUCCESS))
        audit.log(_make_event(status=TransferStatus.DUPLICATE))
        audit.log(_make_event(status=TransferStatus.ERROR))

        entries = audit.read_entries(status=TransferStatus.DUPLICATE)
        assert len(entries) == 1
        assert entries[0].transfer_status == TransferStatus.DUPLICATE

    def test_limit(self, tmp_path):
        audit = WatchdogAuditLog(tmp_path / "audit.jsonl")
        for i in range(5):
            audit.log(_make_event(file_name=f"doc{i}.pdf"))

        entries = audit.read_entries(limit=2)
        assert len(entries) == 2
        assert entries[0].file_name == "doc3.pdf"
        assert entries[1].file_name == "doc4.pdf"

    def test_combined_filters(self, tmp_path):
        audit = WatchdogAuditLog(tmp_path / "audit.jsonl")
        now = datetime.now(UTC)

        old_event = _make_event(file_name="old.pdf")
        old_event.timestamp = now - timedelta(seconds=10)
        audit.log(old_event)

        cutoff = now - timedelta(seconds=5)

        new1 = _make_event(file_name="new1.pdf", status=TransferStatus.SUCCESS)
        new1.timestamp = now - timedelta(seconds=2)
        audit.log(new1)

        new2 = _make_event(file_name="new2.pdf", status=TransferStatus.DUPLICATE)
        new2.timestamp = now - timedelta(seconds=1)
        audit.log(new2)

        new3 = _make_event(file_name="new3.pdf", status=TransferStatus.SUCCESS)
        new3.timestamp = now
        audit.log(new3)

        entries = audit.read_entries(since=cutoff, status=TransferStatus.SUCCESS)
        assert len(entries) == 2
        assert entries[0].file_name == "new1.pdf"
        assert entries[1].file_name == "new3.pdf"


class TestEdgeCases:
    def test_creates_parent_dirs(self, tmp_path):
        audit = WatchdogAuditLog(tmp_path / "deep" / "nested" / "audit.jsonl")
        audit.log(_make_event())
        assert audit.read_entries()[0].file_name == "scan.pdf"

    def test_append_across_instances(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"

        audit1 = WatchdogAuditLog(log_path)
        audit1.log(_make_event(file_name="a.pdf"))

        audit2 = WatchdogAuditLog(log_path)
        audit2.log(_make_event(file_name="b.pdf"))

        entries = audit2.read_entries()
        assert len(entries) == 2


class TestPurge:
    def test_purge_removes_old_keeps_recent(self, tmp_path):
        audit = WatchdogAuditLog(tmp_path / "audit.jsonl")
        now = datetime.now(UTC)

        old_event = _make_event(file_name="old.pdf")
        old_event.timestamp = now - timedelta(days=100)
        audit.log(old_event)

        new_event = _make_event(file_name="new.pdf")
        new_event.timestamp = now
        audit.log(new_event)

        cutoff = now - timedelta(days=50)
        purged = audit.purge_before(cutoff)
        assert purged == 1

        remaining = audit.read_entries()
        assert len(remaining) == 1
        assert remaining[0].file_name == "new.pdf"

    def test_purge_returns_correct_count(self, tmp_path):
        audit = WatchdogAuditLog(tmp_path / "audit.jsonl")
        now = datetime.now(UTC)

        for i in range(4):
            event = _make_event(file_name=f"old{i}.pdf")
            event.timestamp = now - timedelta(days=100)
            audit.log(event)

        new_event = _make_event(file_name="recent.pdf")
        new_event.timestamp = now
        audit.log(new_event)

        purged = audit.purge_before(now - timedelta(days=50))
        assert purged == 4

    def test_purge_noop_when_nothing_old(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        audit = WatchdogAuditLog(log_path)
        audit.log(_make_event())

        old_mtime = log_path.stat().st_mtime

        purged = audit.purge_before(datetime.now(UTC) - timedelta(days=999))
        assert purged == 0
        assert log_path.stat().st_mtime == old_mtime

    def test_purge_nonexistent_file(self, tmp_path):
        audit = WatchdogAuditLog(tmp_path / "nonexistent.jsonl")
        assert audit.purge_before(datetime.now(UTC)) == 0
