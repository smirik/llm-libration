"""Tests for the main CLI interface."""

import pytest
from click.testing import CliRunner

from llm_libration.cli import main


class TestMainCLI:
    """Test cases for main CLI functionality."""

    @pytest.fixture
    def runner(self):
        """Create a Click test runner."""
        return CliRunner()

    def test_main_help(self, runner):
        """Test the main CLI help command."""
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'LLM Libration: AI-powered analysis' in result.output
        assert 'plot' in result.output
        assert 'run' in result.output
        assert 'benchmark' in result.output

    def test_main_version(self, runner):
        """Test the version command."""
        result = runner.invoke(main, ['--version'])
        assert result.exit_code == 0
