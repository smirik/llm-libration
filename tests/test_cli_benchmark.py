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
        with patch('llm_libration.cli.LibrationAnalyzer') as mock_analyzer_class:
            mock_analyzer = MagicMock()
            # Mock perfect classification - files are processed in alphabetical order
            mock_analyzer.analyze_image.side_effect = [
                ResonanceType.NON_RESONANT,  # circulation/circulation_1.png
                ResonanceType.NON_RESONANT,  # circulation/circulation_2.png
                ResonanceType.CONTROVERSIAL,  # controversial/controversial_1.png
                ResonanceType.RESONANT,  # libration/libration_1.png
                ResonanceType.RESONANT,  # libration/libration_2.png
                ResonanceType.CONTROVERSIAL,  # transient/transient_1.png
            ]
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, ['benchmark', str(benchmark_directory), '--provider', 'openai'])

            assert result.exit_code == 0
            assert 'Starting benchmark evaluation' in result.output
            assert 'Found 6 PNG files to analyze' in result.output
            assert 'BENCHMARK RESULTS' in result.output
            assert 'Accuracy:  1.000' in result.output
            assert 'Precision: 1.000' in result.output
            assert 'Recall:    1.000' in result.output
            assert 'F1 Score:  1.000' in result.output

    def test_benchmark_with_custom_model(self, runner, benchmark_directory):
        """Test benchmark command with custom model."""
        with patch('llm_libration.cli.LibrationAnalyzer') as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.analyze_image.return_value = ResonanceType.RESONANT
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, ['benchmark', str(benchmark_directory), '--provider', 'anthropic', '--model', 'claude-3-sonnet'])

            assert result.exit_code == 0
            assert 'Provider: anthropic' in result.output
            assert 'Model: claude-3-sonnet' in result.output
            mock_analyzer_class.assert_called_once_with(provider='anthropic', model_name='claude-3-sonnet')

    def test_benchmark_with_errors(self, runner, benchmark_directory):
        """Test benchmark execution with some analysis errors."""
        with patch('llm_libration.cli.LibrationAnalyzer') as mock_analyzer_class:
            mock_analyzer = MagicMock()
            # Mock mixed results with some errors
            mock_analyzer.analyze_image.side_effect = [
                ResonanceType.NON_RESONANT,  # circulation_1.png
                Exception("Analysis error"),  # circulation_2.png - error
                ResonanceType.CONTROVERSIAL,  # controversial_1.png
                ResonanceType.RESONANT,  # libration_1.png
                ResonanceType.RESONANT,  # libration_2.png
                ResonanceType.CONTROVERSIAL,  # transient_1.png
            ]
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, ['benchmark', str(benchmark_directory), '--provider', 'anthropic'])

            assert result.exit_code == 0
            assert 'Starting benchmark evaluation' in result.output
            assert 'Found 6 PNG files to analyze' in result.output
            assert 'Successful: 5' in result.output
            assert 'Errors: 1' in result.output
            assert 'CLASSIFICATION METRICS' in result.output

    def test_benchmark_analyzer_initialization_error(self, runner, benchmark_directory):
        """Test benchmark command when analyzer initialization fails."""
        with patch('llm_libration.cli.LibrationAnalyzer') as mock_analyzer_class:
            mock_analyzer_class.side_effect = Exception("API key not found")

            result = runner.invoke(main, ['benchmark', str(benchmark_directory)])

            assert result.exit_code == 1
            assert 'Failed to initialize analyzer' in result.output

    def test_benchmark_all_errors(self, runner, benchmark_directory):
        """Test benchmark execution where all analyses fail."""
        with patch('llm_libration.cli.LibrationAnalyzer') as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.analyze_image.side_effect = Exception("Analysis error")
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, ['benchmark', str(benchmark_directory)])

            assert result.exit_code == 0
            assert 'Successful: 0' in result.output
            assert 'Errors: 6' in result.output
            assert 'No successful analyses to calculate metrics' in result.output

    def test_benchmark_provider_options(self, runner, benchmark_directory):
        """Test benchmark command with different provider options."""
        with patch('llm_libration.cli.LibrationAnalyzer') as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.analyze_image.return_value = ResonanceType.RESONANT
            mock_analyzer_class.return_value = mock_analyzer

            # Test each provider option
            providers = ['openai', 'anthropic', 'openrouter', 'ollama']
            for provider in providers:
                result = runner.invoke(main, ['benchmark', str(benchmark_directory), '--provider', provider])
                assert result.exit_code == 0
                assert f'Provider: {provider}' in result.output

    def test_benchmark_mixed_results(self, runner, benchmark_directory):
        """Test benchmark with mixed classification results."""
        with patch('llm_libration.cli.LibrationAnalyzer') as mock_analyzer_class:
            mock_analyzer = MagicMock()
            # Mock mixed results - some correct, some incorrect
            mock_analyzer.analyze_image.side_effect = [
                ResonanceType.RESONANT,  # circulation_1.png (should be non-resonant) - FP
                ResonanceType.NON_RESONANT,  # circulation_2.png (correct) - TN
                ResonanceType.RESONANT,  # controversial_1.png (should be controversial) - FP
                ResonanceType.NON_RESONANT,  # libration_1.png (should be resonant) - FN
                ResonanceType.RESONANT,  # libration_2.png (correct) - TP
                ResonanceType.NON_RESONANT,  # transient_1.png (should be controversial) - FN
            ]
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, ['benchmark', str(benchmark_directory)])

            assert result.exit_code == 0
            assert 'CLASSIFICATION METRICS' in result.output
            assert 'True Positives:    1' in result.output
            assert 'True Negatives:    1' in result.output
            assert 'False Positives:   2' in result.output
            assert 'False Negatives:   2' in result.output

            # Check accuracy calculation: (TP + TN) / Total = (1 + 1) / 6 = 0.333
            assert 'Accuracy:  0.333' in result.output

    def test_benchmark_file_save_error(self, runner, benchmark_directory):
        """Test benchmark with file save error (should show warning but continue)."""
        with patch('llm_libration.cli.LibrationAnalyzer') as mock_analyzer_class, patch(
            'llm_libration.cli.save_benchmark_results'
        ) as mock_save:

            mock_analyzer = MagicMock()
            mock_analyzer.analyze_image.return_value = ResonanceType.RESONANT
            mock_analyzer_class.return_value = mock_analyzer

            mock_save.side_effect = Exception("Permission denied")

            result = runner.invoke(main, ['benchmark', str(benchmark_directory)])

            assert result.exit_code == 0
            assert 'Warning: Failed to save results' in result.output
            assert 'BENCHMARK RESULTS' in result.output  # Should still show results

    def test_benchmark_with_nested_directories(self, runner):
        """Test benchmark with nested directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create nested structure
            nested_lib = temp_path / "results" / "libration" / "subset1"
            nested_lib.mkdir(parents=True)

            # Create PNG files in nested structure
            png_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x18\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82'
            (nested_lib / "test.png").write_bytes(png_content)

            with patch('llm_libration.cli.LibrationAnalyzer') as mock_analyzer_class:
                mock_analyzer = MagicMock()
                mock_analyzer.analyze_image.return_value = ResonanceType.RESONANT
                mock_analyzer_class.return_value = mock_analyzer

                result = runner.invoke(main, ['benchmark', str(temp_path)])

                assert result.exit_code == 0
                assert 'Found 1 PNG files to analyze' in result.output
