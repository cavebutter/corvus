"""Tests for the `corvus watch` CLI command."""

from unittest.mock import patch

import click.testing
import pytest

from corvus.cli import cli


@pytest.fixture
def runner():
    return click.testing.CliRunner()


class TestWatchHelp:
    def test_help_shows_options(self, runner):
        result = runner.invoke(cli, ["watch", "--help"])
        assert result.exit_code == 0
        assert "--scan-dir" in result.output
        assert "--method" in result.output
        assert "--consume-dir" in result.output
        assert "--patterns" in result.output
        assert "--once" in result.output

    def test_method_choices(self, runner):
        result = runner.invoke(cli, ["watch", "--help"])
        assert "move" in result.output
        assert "upload" in result.output


class TestWatchValidation:
    def test_missing_scan_dir(self, runner):
        result = runner.invoke(cli, ["watch", "--once", "--scan-dir", ""])
        assert result.exit_code != 0
        assert "scan-dir" in result.output.lower() or "scan" in result.output.lower()

    def test_nonexistent_scan_dir(self, runner):
        result = runner.invoke(cli, ["watch", "--once", "--scan-dir", "/nonexistent/path"])
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_move_requires_consume_dir(self, runner, tmp_path):
        scan_dir = tmp_path / "scans"
        scan_dir.mkdir()
        result = runner.invoke(
            cli,
            [
                "watch", "--once",
                "--scan-dir", str(scan_dir),
                "--method", "move", "--consume-dir", "",
            ],
        )
        assert result.exit_code != 0
        assert "consume-dir" in result.output.lower() or "consume" in result.output.lower()

    def test_move_nonexistent_consume_dir(self, runner, tmp_path):
        scan_dir = tmp_path / "scans"
        scan_dir.mkdir()
        result = runner.invoke(
            cli,
            [
                "watch", "--once",
                "--scan-dir", str(scan_dir),
                "--method", "move",
                "--consume-dir", "/nonexistent/consume",
            ],
        )
        assert result.exit_code != 0
        assert "does not exist" in result.output


class TestWatchOnce:
    @patch("corvus.cli.WATCHDOG_HASH_DB_PATH", new="")
    @patch("corvus.cli.WATCHDOG_AUDIT_LOG_PATH", new="")
    def test_once_processes_files(self, runner, tmp_path):
        scan_dir = tmp_path / "scans"
        consume_dir = tmp_path / "consume"
        scan_dir.mkdir()
        consume_dir.mkdir()

        # Create test files
        (scan_dir / "doc1.pdf").write_bytes(b"pdf content 1")
        (scan_dir / "doc2.pdf").write_bytes(b"pdf content 2")
        (scan_dir / "notes.txt").write_bytes(b"not a pdf")  # Should be skipped

        hash_db = str(tmp_path / "hashes.db")
        audit_log = str(tmp_path / "audit.jsonl")

        with (
            patch("corvus.cli.WATCHDOG_HASH_DB_PATH", hash_db),
            patch("corvus.cli.WATCHDOG_AUDIT_LOG_PATH", audit_log),
        ):
            result = runner.invoke(
                cli,
                [
                    "watch", "--once",
                    "--scan-dir", str(scan_dir),
                    "--method", "move",
                    "--consume-dir", str(consume_dir),
                    "--patterns", "*.pdf",
                ],
            )

        assert result.exit_code == 0
        assert "Transferred: 2" in result.output
        assert (consume_dir / "doc1.pdf").exists()
        assert (consume_dir / "doc2.pdf").exists()

    @patch("corvus.cli.WATCHDOG_HASH_DB_PATH", new="")
    @patch("corvus.cli.WATCHDOG_AUDIT_LOG_PATH", new="")
    def test_once_empty_dir(self, runner, tmp_path):
        scan_dir = tmp_path / "scans"
        consume_dir = tmp_path / "consume"
        scan_dir.mkdir()
        consume_dir.mkdir()

        hash_db = str(tmp_path / "hashes.db")
        audit_log = str(tmp_path / "audit.jsonl")

        with (
            patch("corvus.cli.WATCHDOG_HASH_DB_PATH", hash_db),
            patch("corvus.cli.WATCHDOG_AUDIT_LOG_PATH", audit_log),
        ):
            result = runner.invoke(
                cli,
                [
                    "watch", "--once",
                    "--scan-dir", str(scan_dir),
                    "--method", "move",
                    "--consume-dir", str(consume_dir),
                ],
            )

        assert result.exit_code == 0
        assert "Files: 0" in result.output

    @patch("corvus.cli.WATCHDOG_HASH_DB_PATH", new="")
    @patch("corvus.cli.WATCHDOG_AUDIT_LOG_PATH", new="")
    def test_once_dedup_on_second_run(self, runner, tmp_path):
        scan_dir = tmp_path / "scans"
        consume_dir = tmp_path / "consume"
        scan_dir.mkdir()
        consume_dir.mkdir()

        (scan_dir / "doc.pdf").write_bytes(b"pdf content")

        hash_db = str(tmp_path / "hashes.db")
        audit_log = str(tmp_path / "audit.jsonl")

        with (
            patch("corvus.cli.WATCHDOG_HASH_DB_PATH", hash_db),
            patch("corvus.cli.WATCHDOG_AUDIT_LOG_PATH", audit_log),
        ):
            # First run — transfers
            result1 = runner.invoke(
                cli,
                [
                    "watch", "--once",
                    "--scan-dir", str(scan_dir),
                    "--method", "move",
                    "--consume-dir", str(consume_dir),
                    "--patterns", "*.pdf",
                ],
            )
            assert "Transferred: 1" in result1.output

            # Put a file with same content back
            (scan_dir / "doc.pdf").write_bytes(b"pdf content")

            # Second run — duplicate detected
            result2 = runner.invoke(
                cli,
                [
                    "watch", "--once",
                    "--scan-dir", str(scan_dir),
                    "--method", "move",
                    "--consume-dir", str(consume_dir),
                    "--patterns", "*.pdf",
                ],
            )
            assert "Duplicates: 1" in result2.output
            assert "Transferred: 0" in result2.output
