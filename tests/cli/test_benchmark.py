"""Tests for CLI benchmark command."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from click.testing import CliRunner

from llm_libration.cli import main
from llm_libration.types import ResonanceType


class TestBenchmarkCLI:
    """Test cases for benchmark CLI command."""

    @pytest.fixture
    def runner(self):
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def benchmark_directory(self):
        """Create a temporary benchmark directory structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create benchmark subdirectories
            (temp_path / "libration").mkdir()
            (temp_path / "circulation").mkdir()
            (temp_path / "transient").mkdir()
            (temp_path / "controversial").mkdir()

            # Create sample PNG files in each category
            png_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x18\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82'

            # Libration category (resonant)
            (temp_path / "libration" / "libration_1.png").write_bytes(png_content)
            (temp_path / "libration" / "libration_2.png").write_bytes(png_content)

            # Circulation category (non-resonant)
            (temp_path / "circulation" / "circulation_1.png").write_bytes(png_content)
            (temp_path / "circulation" / "circulation_2.png").write_bytes(png_content)

            # Transient category (controversial)
            (temp_path / "transient" / "transient_1.png").write_bytes(png_content)

            # Controversial category (controversial)
            (temp_path / "controversial" / "controversial_1.png").write_bytes(png_content)

            yield temp_path

    def test_benchmark_help(self, runner):
        """Test the benchmark command help."""
        result = runner.invoke(main, ['benchmark', '--help'])
        assert result.exit_code == 0
        assert 'Run benchmark evaluation' in result.output
        assert '--provider' in result.output
        assert '--model' in result.output

    def test_benchmark_no_images(self, runner):
        """Test benchmark command with directory containing no PNG files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(main, ['benchmark', temp_dir])
            assert result.exit_code == 1
            assert 'No PNG files found' in result.output

    def test_benchmark_nonexistent_directory(self, runner):
        """Test benchmark command with non-existent directory."""
        result = runner.invoke(main, ['benchmark', '/nonexistent/directory'])
        assert result.exit_code == 2  # Click's file not found exit code

    def test_benchmark_success(self, runner, benchmark_directory):
        """Test successful benchmark execution."""
        with patch('llm_libration.cli.benchmark.LibrationAnalyzer') as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.get_resonance_type.return_value = ResonanceType.RESONANT
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, ['benchmark', str(benchmark_directory)])

            assert result.exit_code == 0
            assert 'BENCHMARK RESULTS' in result.output

    def test_benchmark_with_custom_model(self, runner, benchmark_directory):
        """Test benchmark command with custom model."""
        with patch('llm_libration.cli.benchmark.LibrationAnalyzer') as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.get_resonance_type.side_effect = [
                ResonanceType.RESONANT,
                ResonanceType.NON_RESONANT,
            ]
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, ['benchmark', str(benchmark_directory), '--provider', 'anthropic', '--model', 'claude-sonnet-4'])

            assert result.exit_code == 0
            mock_analyzer_class.assert_called_once_with(provider='anthropic', model_name='claude-sonnet-4')

    def test_benchmark_with_errors(self, runner, benchmark_directory):
        """Test benchmark command with some analysis errors."""
        with patch('llm_libration.cli.benchmark.LibrationAnalyzer') as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.get_resonance_type.side_effect = Exception("Analysis error")
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, ['benchmark', str(benchmark_directory)])

            assert result.exit_code == 0  # Should complete despite errors
            assert 'Errors: 6' in result.output  # 6 files in benchmark_directory

    def test_benchmark_analyzer_initialization_error(self, runner, benchmark_directory):
        """Test benchmark command with analyzer initialization error."""
        with patch('llm_libration.cli.benchmark.LibrationAnalyzer') as mock_analyzer_class:
            mock_analyzer_class.side_effect = Exception("Initialization error")

            result = runner.invoke(main, ['benchmark', str(benchmark_directory)])

            assert result.exit_code == 1
            assert 'Failed to initialize analyzer' in result.output

    def test_benchmark_all_errors(self, runner, benchmark_directory):
        """Test benchmark command where all analyses fail."""
        with patch('llm_libration.cli.benchmark.LibrationAnalyzer') as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.get_resonance_type.side_effect = Exception("Analysis error")
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, ['benchmark', str(benchmark_directory)])

            assert result.exit_code == 0
            assert 'No successful analyses to calculate metrics' in result.output

    def test_benchmark_provider_options(self, runner, benchmark_directory):
        """Test benchmark command with different provider options."""
        providers = ['openai', 'anthropic', 'openrouter', 'ollama']

        for provider in providers:
            with patch('llm_libration.cli.benchmark.LibrationAnalyzer') as mock_analyzer_class:
                mock_analyzer = MagicMock()
                mock_analyzer.get_resonance_type.return_value = ResonanceType.RESONANT
                mock_analyzer_class.return_value = mock_analyzer

                result = runner.invoke(main, ['benchmark', str(benchmark_directory), '--provider', provider])

                assert result.exit_code == 0
                mock_analyzer_class.assert_called_once_with(provider=provider)

    def test_benchmark_mixed_results(self, runner, benchmark_directory):
        """Test benchmark command with mixed success/error results."""
        with patch('llm_libration.cli.benchmark.LibrationAnalyzer') as mock_analyzer_class:
            mock_analyzer = MagicMock()
            # First call succeeds, second fails
            mock_analyzer.get_resonance_type.side_effect = [ResonanceType.RESONANT, Exception("Error")]
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, ['benchmark', str(benchmark_directory)])

            assert result.exit_code == 0
            assert 'Successful: 1' in result.output
            assert 'Errors: 5' in result.output

    def test_benchmark_file_save_error(self, runner, benchmark_directory):
        """Test benchmark command with file save error."""
        with patch('llm_libration.cli.benchmark.LibrationAnalyzer') as mock_analyzer_class, patch(
            'llm_libration.cli.benchmark.save_benchmark_results'
        ) as mock_save:
            mock_analyzer = MagicMock()
            mock_analyzer.get_resonance_type.return_value = ResonanceType.RESONANT
            mock_analyzer_class.return_value = mock_analyzer
            mock_save.side_effect = Exception("Save error")

            result = runner.invoke(main, ['benchmark', str(benchmark_directory)])

            assert result.exit_code == 0  # Should complete despite save error
            assert 'Warning: Failed to save results' in result.output

    def test_benchmark_with_nested_directories(self, runner):
        """Test benchmark command with nested directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested structure
            resonant_dir = Path(temp_dir) / 'resonant'
            resonant_dir.mkdir()
            (resonant_dir / 'test1.png').write_bytes(b'fake png data')

            non_resonant_dir = Path(temp_dir) / 'non-resonant'
            non_resonant_dir.mkdir()
            (non_resonant_dir / 'test2.png').write_bytes(b'fake png data')

            with patch('llm_libration.cli.benchmark.LibrationAnalyzer') as mock_analyzer_class:
                mock_analyzer = MagicMock()
                mock_analyzer.get_resonance_type.return_value = ResonanceType.RESONANT
                mock_analyzer_class.return_value = mock_analyzer

                result = runner.invoke(main, ['benchmark', temp_dir])

                assert result.exit_code == 0
                assert 'Found 2 PNG files to analyze' in result.output
