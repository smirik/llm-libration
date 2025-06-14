"""Tests for the plot CLI command."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
from click.testing import CliRunner
import numpy as np

from llm_libration.cli import main


class TestPlotCLI:
    """Test cases for plot CLI command."""

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
        with patch('llm_libration.cli.plot.create_plots_from_input') as mock_create_plots:
            mock_create_plots.return_value = '/path/to/output.png'

            result = runner.invoke(main, ['plot', sample_csv_file])

            assert result.exit_code == 0
            assert 'Plot created successfully' in result.output
            assert '/path/to/output.png' in result.output
            mock_create_plots.assert_called_once()

    def test_plot_with_options(self, runner, sample_csv_file):
        """Test plot command with custom options."""
        with patch('llm_libration.cli.plot.create_plots_from_input') as mock_create_plots:
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
                input_path=Path(sample_csv_file),
                x_column='time',
                y_column='data',
                output_file=Path('custom.png'),
                x_min=0.0,
                x_max=100000.0,
                y_min=-1.0,
                y_max=10.0,
            )

    def test_plot_file_not_found(self, runner):
        """Test plot command with non-existent file."""
        result = runner.invoke(main, ['plot', 'nonexistent.csv'])
        assert result.exit_code == 2  # Click's file not found exit code

    def test_plot_error_handling(self, runner, sample_csv_file):
        """Test plot command error handling."""
        with patch('llm_libration.cli.plot.create_plots_from_input') as mock_create_plots:
            mock_create_plots.side_effect = ValueError("Test error")

            result = runner.invoke(main, ['plot', sample_csv_file])

            assert result.exit_code == 1
            assert 'Error creating plot(s): Test error' in result.output

    def test_plot_folder_processing(self, runner):
        """Test plot command with folder input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('llm_libration.cli.plot.create_plots_from_input') as mock_create_plots:
                mock_create_plots.return_value = ['/path/to/plot1.png', '/path/to/plot2.png']

                result = runner.invoke(main, ['plot', temp_dir])

                assert result.exit_code == 0
                assert 'Batch processing completed' in result.output
                assert '2 plots created' in result.output
                mock_create_plots.assert_called_once_with(
                    input_path=Path(temp_dir),
                    x_column='times',
                    y_column='angle',
                    output_file=None,
                    x_min=0.0,
                    x_max=100000.0,
                    y_min=0.0,
                    y_max=2 * np.pi,
                )
