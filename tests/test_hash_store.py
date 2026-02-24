"""Tests for the SQLite-backed SHA-256 hash store."""

import sqlite3

import pytest

from corvus.watchdog.hash_store import HashStore


class TestBasicOperations:
    def test_add_and_contains(self, tmp_path):
        with HashStore(tmp_path / "hashes.db") as store:
            assert store.contains("abc123") is False
            store.add("abc123", "/scans/doc.pdf")
            assert store.contains("abc123") is True

    def test_get_returns_record(self, tmp_path):
        with HashStore(tmp_path / "hashes.db") as store:
            store.add("abc123", "/scans/doc.pdf")
            record = store.get("abc123")
            assert record is not None
            assert record["sha256"] == "abc123"
            assert record["source_path"] == "/scans/doc.pdf"
            assert record["file_name"] == "doc.pdf"
            assert record["processed_at"] is not None

    def test_get_missing_returns_none(self, tmp_path):
        with HashStore(tmp_path / "hashes.db") as store:
            assert store.get("nonexistent") is None

    def test_count(self, tmp_path):
        with HashStore(tmp_path / "hashes.db") as store:
            assert store.count() == 0
            store.add("hash1", "/scans/a.pdf")
            store.add("hash2", "/scans/b.pdf")
            assert store.count() == 2

    def test_remove(self, tmp_path):
        with HashStore(tmp_path / "hashes.db") as store:
            store.add("abc123", "/scans/doc.pdf")
            assert store.remove("abc123") is True
            assert store.contains("abc123") is False
            assert store.count() == 0

    def test_remove_nonexistent_returns_false(self, tmp_path):
        with HashStore(tmp_path / "hashes.db") as store:
            assert store.remove("nonexistent") is False


class TestDuplicateDetection:
    def test_add_duplicate_raises(self, tmp_path):
        with HashStore(tmp_path / "hashes.db") as store:
            store.add("abc123", "/scans/doc.pdf")
            with pytest.raises(sqlite3.IntegrityError):
                store.add("abc123", "/scans/doc2.pdf")

    def test_different_hashes_same_path_ok(self, tmp_path):
        with HashStore(tmp_path / "hashes.db") as store:
            store.add("hash1", "/scans/doc.pdf")
            store.add("hash2", "/scans/doc.pdf")
            assert store.count() == 2


class TestPersistence:
    def test_data_survives_reopen(self, tmp_path):
        db_path = tmp_path / "hashes.db"

        store1 = HashStore(db_path)
        store1.add("abc123", "/scans/doc.pdf")
        store1.close()

        store2 = HashStore(db_path)
        assert store2.contains("abc123") is True
        assert store2.count() == 1
        store2.close()

    def test_creates_parent_dirs(self, tmp_path):
        db_path = tmp_path / "deep" / "nested" / "hashes.db"
        with HashStore(db_path) as store:
            store.add("abc123", "/scans/doc.pdf")
            assert store.contains("abc123") is True
