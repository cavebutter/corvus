"""Tests for the corvus voice CLI command."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from corvus.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestVoiceCli:
    def test_missing_deps_error(self, runner):
        """Test that missing voice deps gives a clear error."""
        with patch("corvus.cli._check_voice_deps") as mock_check:
            mock_check.side_effect = SystemExit(1)
            result = runner.invoke(cli, ["voice"])
            assert result.exit_code != 0

    def test_no_wakeword_flag_accepted(self, runner):
        """Test that --no-wakeword flag is accepted."""
        with patch("corvus.cli._check_voice_deps"):
            with patch("corvus.cli._validate_config"):
                with patch("corvus.cli.asyncio.run") as mock_run:
                    result = runner.invoke(cli, ["voice", "--no-wakeword"])
                    mock_run.assert_called_once()

    def test_voice_name_option(self, runner):
        """Test that --voice option is accepted."""
        with patch("corvus.cli._check_voice_deps"):
            with patch("corvus.cli._validate_config"):
                with patch("corvus.cli.asyncio.run") as mock_run:
                    result = runner.invoke(cli, ["voice", "--voice", "bf_emma"])
                    mock_run.assert_called_once()

    def test_new_flag(self, runner):
        """Test that --new flag is accepted."""
        with patch("corvus.cli._check_voice_deps"):
            with patch("corvus.cli._validate_config"):
                with patch("corvus.cli.asyncio.run") as mock_run:
                    result = runner.invoke(cli, ["voice", "--new"])
                    mock_run.assert_called_once()

    def test_resume_option(self, runner):
        """Test that --resume option is accepted."""
        with patch("corvus.cli._check_voice_deps"):
            with patch("corvus.cli._validate_config"):
                with patch("corvus.cli.asyncio.run") as mock_run:
                    result = runner.invoke(cli, ["voice", "--resume", "abc123"])
                    mock_run.assert_called_once()

    def test_help_shows_voice_command(self, runner):
        """Test that voice command appears in help output."""
        result = runner.invoke(cli, ["--help"])
        assert "voice" in result.output

    def test_voice_help(self, runner):
        """Test that voice --help works."""
        result = runner.invoke(cli, ["voice", "--help"])
        assert result.exit_code == 0
        assert "wake word" in result.output.lower() or "speech" in result.output.lower()
