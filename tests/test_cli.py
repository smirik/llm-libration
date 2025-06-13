"""Tests for the Command Line Interface."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from click.testing import CliRunner
import numpy as np

from llm_libration.cli import (
    main,
    plot,
    run,
    analyze_multiple_images,
)
from llm_libration.types import ResonanceType


class TestCLI:
    """Test cases for CLI functionality."""

    @pytest.fixture
    def runner(self):
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('times,angle\n')
            f.write('0.0,0.0\n')
            f.write('1.0,1.57\n')
            f.write('2.0,3.14\n')
            f.write('3.0,4.71\n')
            f.write('4.0,6.28\n')
            csv_path = f.name

        yield csv_path

        # Cleanup
        os.unlink(csv_path)

    @pytest.fixture
    def sample_image_file(self):
        """Create a temporary PNG file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            # Write a minimal valid PNG header
            f.write(
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x18\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82'
            )
            image_path = f.name

        yield image_path

        # Cleanup
        os.unlink(image_path)

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

    def test_plot_help(self, runner):
        """Test the plot command help."""
        result = runner.invoke(main, ['plot', '--help'])
        assert result.exit_code == 0
        assert 'Create plot(s) from CSV data' in result.output
        assert '--x-column' in result.output
        assert '--y-column' in result.output
        assert '--output-file' in result.output

    def test_plot_success(self, runner, sample_csv_file):
        """Test successful plot creation."""
        with patch('llm_libration.cli.create_plots_from_input') as mock_create_plots:
            mock_create_plots.return_value = '/path/to/output.png'

            result = runner.invoke(main, ['plot', sample_csv_file])

            assert result.exit_code == 0
            assert 'Plot created successfully' in result.output
            assert '/path/to/output.png' in result.output
            mock_create_plots.assert_called_once()

    def test_plot_with_options(self, runner, sample_csv_file):
        """Test plot command with custom options."""
        with patch('llm_libration.cli.create_plots_from_input') as mock_create_plots:
            mock_create_plots.return_value = '/path/to/custom.png'

            result = runner.invoke(
                main,
                [
                    'plot',
                    sample_csv_file,
                    '--x-column',
                    'time',
                    '--y-column',
                    'data',
                    '--output-file',
                    'custom.png',
                    '--y-min',
                    '-1.0',
                    '--y-max',
                    '10.0',
                ],
            )

            assert result.exit_code == 0
            mock_create_plots.assert_called_once_with(
                input_path=Path(sample_csv_file), x_column='time', y_column='data', output_file=Path('custom.png'), y_min=-1.0, y_max=10.0
            )

    def test_plot_file_not_found(self, runner):
        """Test plot command with non-existent file."""
        result = runner.invoke(main, ['plot', 'nonexistent.csv'])
        assert result.exit_code == 2  # Click's file not found exit code

    def test_plot_error_handling(self, runner, sample_csv_file):
        """Test plot command error handling."""
        with patch('llm_libration.cli.create_plots_from_input') as mock_create_plots:
            mock_create_plots.side_effect = ValueError("Test error")

            result = runner.invoke(main, ['plot', sample_csv_file])

            assert result.exit_code == 1
            assert 'Error creating plot(s): Test error' in result.output

    def test_plot_folder_processing(self, runner):
        """Test plot command with folder input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('llm_libration.cli.create_plots_from_input') as mock_create_plots:
                mock_create_plots.return_value = ['/path/to/plot1.png', '/path/to/plot2.png']

                result = runner.invoke(main, ['plot', temp_dir])

                assert result.exit_code == 0
                assert 'Batch processing completed' in result.output
                assert '2 plots created' in result.output
                mock_create_plots.assert_called_once_with(
                    input_path=Path(temp_dir), x_column='times', y_column='angle', output_file=None, y_min=0.0, y_max=2 * np.pi
                )

    def test_run_help(self, runner):
        """Test the run command help."""
        result = runner.invoke(main, ['run', '--help'])
        assert result.exit_code == 0
        assert 'Run libration analysis' in result.output
        assert '--provider' in result.output
        assert '--model' in result.output

    def test_run_single_image(self, runner, sample_image_file):
        """Test run command with single image."""
        with patch('llm_libration.cli.analyze_multiple_images') as mock_analyze:
            result = runner.invoke(main, ['run', sample_image_file])

            assert result.exit_code == 0
            mock_analyze.assert_called_once_with([Path(sample_image_file)], 'openai', None)

    def test_run_multiple_images(self, runner, sample_image_file):
        """Test run command with multiple images."""
        # Create a second image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x18\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82'
            )
            image_path2 = f.name

        try:
            with patch('llm_libration.cli.analyze_multiple_images') as mock_analyze:
                result = runner.invoke(main, ['run', sample_image_file, image_path2, '--provider', 'anthropic'])

                assert result.exit_code == 0
                mock_analyze.assert_called_once_with([Path(sample_image_file), Path(image_path2)], 'anthropic', None)
        finally:
            os.unlink(image_path2)

    def test_run_with_custom_model(self, runner, sample_image_file):
        """Test run command with custom model."""
        with patch('llm_libration.cli.analyze_multiple_images') as mock_analyze:
            result = runner.invoke(main, ['run', sample_image_file, '--provider', 'openai', '--model', 'gpt-4-vision'])

            assert result.exit_code == 0
            mock_analyze.assert_called_once_with([Path(sample_image_file)], 'openai', 'gpt-4-vision')

    def test_run_error_handling(self, runner, sample_image_file):
        """Test run command error handling."""
        with patch('llm_libration.cli.analyze_multiple_images') as mock_analyze:
            mock_analyze.side_effect = Exception("Analysis failed")

            result = runner.invoke(main, ['run', sample_image_file])

            assert result.exit_code == 1
            assert 'Analysis failed: Analysis failed' in result.output

    def test_run_image_not_found(self, runner):
        """Test run command with non-existent image."""
        result = runner.invoke(main, ['run', 'nonexistent.png'])
        assert result.exit_code == 2  # Click's file not found exit code

    def test_analyze_multiple_images_single_provider(self, sample_image_file):
        """Test analyze_multiple_images function with single provider."""
        with patch('llm_libration.cli.LibrationAnalyzer') as mock_analyzer_class, patch('llm_libration.cli.click.echo') as mock_echo:

            mock_analyzer = MagicMock()
            mock_analyzer.analyze_image.return_value = MagicMock(value='resonant')
            mock_analyzer_class.return_value = mock_analyzer

            analyze_multiple_images([Path(sample_image_file)], 'openai', None)

            mock_analyzer_class.assert_called_once_with(provider='openai')
            mock_analyzer.analyze_image.assert_called_once()

    def test_analyze_multiple_images_all_providers(self, sample_image_file):
        """Test analyze_multiple_images function with all providers."""
        with patch('llm_libration.cli.LibrationAnalyzer') as mock_analyzer_class, patch('llm_libration.cli.click.echo') as mock_echo:

            mock_analyzer = MagicMock()
            mock_analyzer.analyze_image.return_value = MagicMock(value='resonant')
            mock_analyzer_class.return_value = mock_analyzer

            analyze_multiple_images([Path(sample_image_file)], 'all', None)

            # Should be called 4 times (once for each provider)
            assert mock_analyzer_class.call_count == 4
            assert mock_analyzer.analyze_image.call_count == 4

    def test_analyze_multiple_images_with_custom_model(self, sample_image_file):
        """Test analyze_multiple_images function with custom model."""
        with patch('llm_libration.cli.LibrationAnalyzer') as mock_analyzer_class, patch('llm_libration.cli.click.echo') as mock_echo:

            mock_analyzer = MagicMock()
            mock_analyzer.analyze_image.return_value = MagicMock(value='resonant')
            mock_analyzer_class.return_value = mock_analyzer

            analyze_multiple_images([Path(sample_image_file)], 'anthropic', 'claude-sonnet-4')

            mock_analyzer_class.assert_called_once_with(provider='anthropic', model_name='claude-sonnet-4')

    def test_analyze_multiple_images_error_handling(self, sample_image_file):
        """Test analyze_multiple_images function error handling."""
        with patch('llm_libration.cli.LibrationAnalyzer') as mock_analyzer_class, patch('llm_libration.cli.click.echo') as mock_echo:

            mock_analyzer = MagicMock()
            mock_analyzer.analyze_image.side_effect = Exception("Analysis error")
            mock_analyzer_class.return_value = mock_analyzer

            # Should not raise exception, just print error
            analyze_multiple_images([Path(sample_image_file)], 'openai', None)

            # Check that error was echoed
            error_calls = [call for call in mock_echo.call_args_list if 'Error' in str(call)]
            assert len(error_calls) > 0

    def test_analyze_multiple_images_nonexistent_file(self):
        """Test analyze_multiple_images function with non-existent file."""
        with patch('llm_libration.cli.click.echo') as mock_echo:

            analyze_multiple_images([Path('/nonexistent/file.png')], 'openai', None)

            # Check that error was echoed for missing file
            error_calls = [call for call in mock_echo.call_args_list if 'not found' in str(call)]
            assert len(error_calls) > 0
