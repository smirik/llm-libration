"""Tests for the plotting functionality."""

import os
import tempfile
import csv
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import pandas as pd

from llm_libration.data.plot import create_plot, find_csv_files, create_plots_from_folder, create_plots_from_input


class TestCreatePlot:
    """Test cases for create_plot function."""

    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        return {'times': [0.0, 1.0, 2.0, 3.0, 4.0], 'angle': [0.0, 1.57, 3.14, 4.71, 6.28], 'other_column': [10, 20, 30, 40, 50]}

    @pytest.fixture
    def csv_file(self, sample_csv_data):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=sample_csv_data.keys())
            writer.writeheader()
            for i in range(len(sample_csv_data['times'])):
                row = {col: sample_csv_data[col][i] for col in sample_csv_data.keys()}
                writer.writerow(row)
            csv_path = f.name

        yield csv_path

        # Cleanup
        os.unlink(csv_path)

    def test_create_plot_default_parameters(self, csv_file):
        """Test create_plot with default parameters."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig, patch('matplotlib.pyplot.close') as mock_close, patch(
            'matplotlib.pyplot.figure'
        ) as mock_figure, patch('matplotlib.pyplot.plot') as mock_plot, patch('matplotlib.pyplot.xlim') as mock_xlim, patch(
            'matplotlib.pyplot.ylim'
        ) as mock_ylim, patch(
            'matplotlib.pyplot.xticks'
        ) as mock_xticks, patch(
            'matplotlib.pyplot.yticks'
        ) as mock_yticks, patch(
            'matplotlib.pyplot.gca'
        ) as mock_gca:

            # Mock gca to return an object without legend
            mock_axes = MagicMock()
            mock_axes.get_legend.return_value = None
            mock_gca.return_value = mock_axes

            result = create_plot(csv_file)

            # Check that the result is the expected PNG path
            expected_path = str(Path(csv_file).parent / f"{Path(csv_file).stem}.png")
            assert result == str(Path(expected_path).absolute())

            # Verify matplotlib calls
            mock_figure.assert_called_once_with(figsize=(10, 6))
            mock_plot.assert_called_once()
            mock_xlim.assert_called_once_with(0, 100000)
            mock_ylim.assert_called_once_with(0, 2 * np.pi)
            mock_xticks.assert_called_once_with([])
            mock_yticks.assert_called_once_with([])
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()

    def test_create_plot_custom_columns(self, csv_file):
        """Test create_plot with custom column names."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig, patch('matplotlib.pyplot.close') as mock_close, patch(
            'matplotlib.pyplot.figure'
        ) as mock_figure, patch('matplotlib.pyplot.plot') as mock_plot, patch('matplotlib.pyplot.xlim') as mock_xlim, patch(
            'matplotlib.pyplot.ylim'
        ) as mock_ylim, patch(
            'matplotlib.pyplot.xticks'
        ) as mock_xticks, patch(
            'matplotlib.pyplot.yticks'
        ) as mock_yticks, patch(
            'matplotlib.pyplot.gca'
        ) as mock_gca:

            mock_axes = MagicMock()
            mock_axes.get_legend.return_value = None
            mock_gca.return_value = mock_axes

            result = create_plot(csv_file, x_column='times', y_column='other_column')

            # Verify the plot was called with correct data
            mock_plot.assert_called_once()
            # Check that the call included the expected data
            call_args = mock_plot.call_args
            assert len(call_args[0]) == 2  # x and y data

            # Verify other calls
            mock_figure.assert_called_once_with(figsize=(10, 6))
            mock_xlim.assert_called_once_with(0, 100000)
            mock_ylim.assert_called_once_with(0, 2 * np.pi)

    def test_create_plot_custom_output_file(self, csv_file):
        """Test create_plot with custom output file path."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as output_file:
            output_path = output_file.name

        try:
            with patch('matplotlib.pyplot.savefig') as mock_savefig, patch('matplotlib.pyplot.close') as mock_close, patch(
                'matplotlib.pyplot.figure'
            ) as mock_figure, patch('matplotlib.pyplot.plot') as mock_plot, patch('matplotlib.pyplot.xlim') as mock_xlim, patch(
                'matplotlib.pyplot.ylim'
            ) as mock_ylim, patch(
                'matplotlib.pyplot.xticks'
            ) as mock_xticks, patch(
                'matplotlib.pyplot.yticks'
            ) as mock_yticks, patch(
                'matplotlib.pyplot.gca'
            ) as mock_gca:

                mock_axes = MagicMock()
                mock_axes.get_legend.return_value = None
                mock_gca.return_value = mock_axes

                result = create_plot(csv_file, output_file=output_path)

                # Check that the result is the expected PNG path
                assert result == str(Path(output_path).absolute())

                # Verify savefig was called with the custom path
                mock_savefig.assert_called_once_with(Path(output_path), bbox_inches='tight', dpi=150)
        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_create_plot_custom_y_limits(self, csv_file):
        """Test create_plot with custom y-axis limits."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig, patch('matplotlib.pyplot.close') as mock_close, patch(
            'matplotlib.pyplot.figure'
        ) as mock_figure, patch('matplotlib.pyplot.plot') as mock_plot, patch('matplotlib.pyplot.xlim') as mock_xlim, patch(
            'matplotlib.pyplot.ylim'
        ) as mock_ylim, patch(
            'matplotlib.pyplot.xticks'
        ) as mock_xticks, patch(
            'matplotlib.pyplot.yticks'
        ) as mock_yticks, patch(
            'matplotlib.pyplot.gca'
        ) as mock_gca:

            mock_axes = MagicMock()
            mock_axes.get_legend.return_value = None
            mock_gca.return_value = mock_axes

            custom_y_min = -1.0
            custom_y_max = 10.0

            result = create_plot(csv_file, y_min=custom_y_min, y_max=custom_y_max)

            # Verify ylim was called with custom limits
            mock_xlim.assert_called_once_with(0, 100000)
            mock_ylim.assert_called_once_with(custom_y_min, custom_y_max)

    def test_create_plot_custom_x_limits(self, csv_file):
        """Test create_plot with custom x-axis limits."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig, patch('matplotlib.pyplot.close') as mock_close, patch(
            'matplotlib.pyplot.figure'
        ) as mock_figure, patch('matplotlib.pyplot.plot') as mock_plot, patch('matplotlib.pyplot.xlim') as mock_xlim, patch(
            'matplotlib.pyplot.ylim'
        ) as mock_ylim, patch(
            'matplotlib.pyplot.xticks'
        ) as mock_xticks, patch(
            'matplotlib.pyplot.yticks'
        ) as mock_yticks, patch(
            'matplotlib.pyplot.gca'
        ) as mock_gca:

            mock_axes = MagicMock()
            mock_axes.get_legend.return_value = None
            mock_gca.return_value = mock_axes

            custom_x_min = 10.0
            custom_x_max = 50000.0

            result = create_plot(csv_file, x_min=custom_x_min, x_max=custom_x_max)

            # Verify xlim was called with custom limits
            mock_xlim.assert_called_once_with(custom_x_min, custom_x_max)
            mock_ylim.assert_called_once_with(0, 2 * np.pi)

    def test_create_plot_custom_x_and_y_limits(self, csv_file):
        """Test create_plot with custom x and y axis limits."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig, patch('matplotlib.pyplot.close') as mock_close, patch(
            'matplotlib.pyplot.figure'
        ) as mock_figure, patch('matplotlib.pyplot.plot') as mock_plot, patch('matplotlib.pyplot.xlim') as mock_xlim, patch(
            'matplotlib.pyplot.ylim'
        ) as mock_ylim, patch(
            'matplotlib.pyplot.xticks'
        ) as mock_xticks, patch(
            'matplotlib.pyplot.yticks'
        ) as mock_yticks, patch(
            'matplotlib.pyplot.gca'
        ) as mock_gca:

            mock_axes = MagicMock()
            mock_axes.get_legend.return_value = None
            mock_gca.return_value = mock_axes

            custom_x_min = 10.0
            custom_x_max = 50000.0
            custom_y_min = -1.0
            custom_y_max = 10.0

            result = create_plot(csv_file, x_min=custom_x_min, x_max=custom_x_max, y_min=custom_y_min, y_max=custom_y_max)

            # Verify both xlim and ylim were called with custom limits
            mock_xlim.assert_called_once_with(custom_x_min, custom_x_max)
            mock_ylim.assert_called_once_with(custom_y_min, custom_y_max)

    def test_create_plot_file_not_found(self):
        """Test create_plot raises FileNotFoundError for non-existent file."""
        non_existent_file = "/path/to/non/existent/file.csv"

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            create_plot(non_existent_file)

    def test_create_plot_invalid_csv(self):
        """Test create_plot raises ValueError for invalid CSV file with missing columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("This is not a valid CSV file content")
            invalid_csv_path = f.name

        try:
            with pytest.raises(ValueError, match="Column 'times' not found in CSV"):
                create_plot(invalid_csv_path)
        finally:
            os.unlink(invalid_csv_path)

    def test_create_plot_missing_x_column(self, csv_file):
        """Test create_plot raises ValueError for missing x column."""
        with pytest.raises(ValueError, match="Column 'non_existent_x' not found in CSV"):
            create_plot(csv_file, x_column='non_existent_x')

    def test_create_plot_missing_y_column(self, csv_file):
        """Test create_plot raises ValueError for missing y column."""
        with pytest.raises(ValueError, match="Column 'non_existent_y' not found in CSV"):
            create_plot(csv_file, y_column='non_existent_y')

    def test_create_plot_plot_styling(self, csv_file):
        """Test that the plot is created with correct styling parameters."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig, patch('matplotlib.pyplot.close') as mock_close, patch(
            'matplotlib.pyplot.figure'
        ) as mock_figure, patch('matplotlib.pyplot.plot') as mock_plot, patch('matplotlib.pyplot.xlim') as mock_xlim, patch(
            'matplotlib.pyplot.ylim'
        ) as mock_ylim, patch(
            'matplotlib.pyplot.xticks'
        ) as mock_xticks, patch(
            'matplotlib.pyplot.yticks'
        ) as mock_yticks, patch(
            'matplotlib.pyplot.gca'
        ) as mock_gca:

            mock_axes = MagicMock()
            mock_axes.get_legend.return_value = None
            mock_gca.return_value = mock_axes

            create_plot(csv_file)

            # Verify plot was called with correct styling
            mock_plot.assert_called_once()
            call_args, call_kwargs = mock_plot.call_args

            # Check styling parameters
            assert call_kwargs.get('linestyle') == ''
            assert call_kwargs.get('marker') == '.'
            assert call_kwargs.get('color') == 'black'

    def test_create_plot_with_legend_removal(self, csv_file):
        """Test that legend is properly removed when it exists."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig, patch('matplotlib.pyplot.close') as mock_close, patch(
            'matplotlib.pyplot.figure'
        ) as mock_figure, patch('matplotlib.pyplot.plot') as mock_plot, patch('matplotlib.pyplot.xlim') as mock_xlim, patch(
            'matplotlib.pyplot.ylim'
        ) as mock_ylim, patch(
            'matplotlib.pyplot.xticks'
        ) as mock_xticks, patch(
            'matplotlib.pyplot.yticks'
        ) as mock_yticks, patch(
            'matplotlib.pyplot.gca'
        ) as mock_gca:

            # Mock gca to return an object with a legend
            mock_legend = MagicMock()
            mock_axes = MagicMock()
            mock_axes.get_legend.return_value = mock_legend
            mock_axes.legend.return_value = mock_legend
            mock_gca.return_value = mock_axes

            create_plot(csv_file)

            # The actual code calls plt.gca().legend().set_visible(False) if plt.gca().get_legend() else None
            # So we need to verify that legend() was called and set_visible was called
            mock_axes.legend.assert_called_once()
            mock_legend.set_visible.assert_called_once_with(False)

    def test_create_plot_path_handling(self, csv_file):
        """Test that Path objects are handled correctly."""
        csv_path = Path(csv_file)

        with patch('matplotlib.pyplot.savefig') as mock_savefig, patch('matplotlib.pyplot.close') as mock_close, patch(
            'matplotlib.pyplot.figure'
        ) as mock_figure, patch('matplotlib.pyplot.plot') as mock_plot, patch('matplotlib.pyplot.xlim') as mock_xlim, patch(
            'matplotlib.pyplot.ylim'
        ) as mock_ylim, patch(
            'matplotlib.pyplot.xticks'
        ) as mock_xticks, patch(
            'matplotlib.pyplot.yticks'
        ) as mock_yticks, patch(
            'matplotlib.pyplot.gca'
        ) as mock_gca:

            mock_axes = MagicMock()
            mock_axes.get_legend.return_value = None
            mock_gca.return_value = mock_axes

            result = create_plot(csv_path)

            # Check that the result is a string path
            assert isinstance(result, str)
            assert result.endswith('.png')

    def test_create_plot_with_real_data_structure(self):
        """Test create_plot with data structure similar to actual use case."""
        # Create CSV data that mimics resonance angle data
        data = {'times': np.linspace(0, 100, 1000), 'angle': np.sin(np.linspace(0, 4 * np.pi, 1000)) * np.pi + np.pi}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(data)
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            with patch('matplotlib.pyplot.savefig') as mock_savefig, patch('matplotlib.pyplot.close') as mock_close, patch(
                'matplotlib.pyplot.figure'
            ) as mock_figure, patch('matplotlib.pyplot.plot') as mock_plot, patch('matplotlib.pyplot.xlim') as mock_xlim, patch(
                'matplotlib.pyplot.ylim'
            ) as mock_ylim, patch(
                'matplotlib.pyplot.xticks'
            ) as mock_xticks, patch(
                'matplotlib.pyplot.yticks'
            ) as mock_yticks, patch(
                'matplotlib.pyplot.gca'
            ) as mock_gca:

                mock_axes = MagicMock()
                mock_axes.get_legend.return_value = None
                mock_gca.return_value = mock_axes

                result = create_plot(csv_path)

                # Verify the plot was created successfully
                assert result.endswith('.png')
                mock_plot.assert_called_once()
                mock_xlim.assert_called_once_with(0, 100000)
                mock_ylim.assert_called_once_with(0, 2 * np.pi)
        finally:
            os.unlink(csv_path)


class TestFolderProcessing:
    """Test cases for folder processing functionality."""

    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        return {'times': [0.0, 1.0, 2.0, 3.0, 4.0], 'angle': [0.0, 1.57, 3.14, 4.71, 6.28]}

    @pytest.fixture
    def csv_file(self, sample_csv_data):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=sample_csv_data.keys())
            writer.writeheader()
            for i in range(len(sample_csv_data['times'])):
                row = {col: sample_csv_data[col][i] for col in sample_csv_data.keys()}
                writer.writerow(row)
            csv_path = f.name

        yield csv_path

        # Cleanup
        os.unlink(csv_path)

    @pytest.fixture
    def temp_folder_with_csv_files(self, sample_csv_data):
        """Create a temporary folder structure with CSV files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create main directory CSV files
            for i in range(2):
                csv_file = temp_path / f"data_{i}.csv"
                df = pd.DataFrame(sample_csv_data)
                df.to_csv(csv_file, index=False)

            # Create subdirectory with CSV files
            subdir = temp_path / "subdir"
            subdir.mkdir()
            for i in range(3):
                csv_file = subdir / f"subdata_{i}.csv"
                df = pd.DataFrame(sample_csv_data)
                df.to_csv(csv_file, index=False)

            # Create a file that's not CSV to ensure it's ignored
            (temp_path / "not_csv.txt").write_text("This is not a CSV file")

            yield temp_path

    def test_find_csv_files(self, temp_folder_with_csv_files):
        """Test find_csv_files function."""
        csv_files = find_csv_files(temp_folder_with_csv_files)

        # Should find 5 CSV files (2 in main dir + 3 in subdir)
        assert len(csv_files) == 5

        # All should be CSV files
        for csv_file in csv_files:
            assert csv_file.suffix == '.csv'
            assert csv_file.is_file()

        # Should be sorted
        assert csv_files == sorted(csv_files)

    def test_find_csv_files_nonexistent_directory(self):
        """Test find_csv_files with non-existent directory."""
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            find_csv_files(Path("/nonexistent/directory"))

    def test_find_csv_files_not_directory(self, csv_file):
        """Test find_csv_files with a file instead of directory."""
        with pytest.raises(NotADirectoryError, match="Path is not a directory"):
            find_csv_files(Path(csv_file))

    def test_find_csv_files_empty_directory(self):
        """Test find_csv_files with directory containing no CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "not_csv.txt").write_text("Not a CSV")

            csv_files = find_csv_files(temp_path)
            assert len(csv_files) == 0

    def test_create_plots_from_folder(self, temp_folder_with_csv_files):
        """Test create_plots_from_folder function."""
        with patch('llm_libration.data.plot.create_plot') as mock_create_plot:
            mock_create_plot.return_value = "/path/to/plot.png"

            result = create_plots_from_folder(temp_folder_with_csv_files, verbose=False)

            # Should have created 5 plots
            assert len(result) == 5

            # create_plot should have been called 5 times
            assert mock_create_plot.call_count == 5

            # All results should be the mocked path
            assert all(path == "/path/to/plot.png" for path in result)

    def test_create_plots_from_folder_with_verbose(self, temp_folder_with_csv_files):
        """Test create_plots_from_folder with verbose output."""
        with patch('llm_libration.data.plot.create_plot') as mock_create_plot, patch('llm_libration.data.plot.click.echo') as mock_echo:

            mock_create_plot.return_value = str(temp_folder_with_csv_files / "plot.png")

            result = create_plots_from_folder(temp_folder_with_csv_files, verbose=True)

            # Should have echoed progress messages
            assert mock_echo.call_count > 0

            # Check that "Found X CSV file(s)" was printed
            found_calls = [call for call in mock_echo.call_args_list if "Found" in str(call)]
            assert len(found_calls) > 0

    def test_create_plots_from_folder_with_errors(self, temp_folder_with_csv_files):
        """Test create_plots_from_folder handles errors gracefully."""
        with patch('llm_libration.data.plot.create_plot') as mock_create_plot:
            # Make some calls succeed and some fail
            mock_create_plot.side_effect = [
                "/path/to/plot1.png",
                Exception("Error creating plot"),
                "/path/to/plot2.png",
                Exception("Another error"),
                "/path/to/plot3.png",
            ]

            result = create_plots_from_folder(temp_folder_with_csv_files, verbose=False)

            # Should return only successful plots
            assert len(result) == 3
            assert all(path.endswith(".png") for path in result)

    def test_create_plots_from_folder_empty_directory(self):
        """Test create_plots_from_folder with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = create_plots_from_folder(temp_dir, verbose=False)
            assert result == []

    def test_create_plots_from_folder_nonexistent_directory(self):
        """Test create_plots_from_folder with non-existent directory."""
        with pytest.raises(FileNotFoundError):
            create_plots_from_folder("/nonexistent/directory")

    def test_create_plots_from_input_single_file(self, csv_file):
        """Test create_plots_from_input with single file."""
        with patch('llm_libration.data.plot.create_plot') as mock_create_plot:
            mock_create_plot.return_value = "/path/to/plot.png"

            result = create_plots_from_input(csv_file)

            assert result == "/path/to/plot.png"
            mock_create_plot.assert_called_once()

    def test_create_plots_from_input_folder(self, temp_folder_with_csv_files):
        """Test create_plots_from_input with folder."""
        with patch('llm_libration.data.plot.create_plots_from_folder') as mock_create_plots:
            mock_create_plots.return_value = ["/path/to/plot1.png", "/path/to/plot2.png"]

            result = create_plots_from_input(temp_folder_with_csv_files)

            assert isinstance(result, list)
            assert len(result) == 2
            mock_create_plots.assert_called_once()

    def test_create_plots_from_input_nonexistent_path(self):
        """Test create_plots_from_input with non-existent path."""
        with pytest.raises(FileNotFoundError, match="Input path not found"):
            create_plots_from_input("/nonexistent/path")

    def test_create_plots_from_input_folder_with_output_file_warning(self, temp_folder_with_csv_files):
        """Test that output_file parameter is ignored for folders with warning."""
        with patch('llm_libration.data.plot.create_plots_from_folder') as mock_create_plots, patch(
            'llm_libration.data.plot.click.echo'
        ) as mock_echo:

            mock_create_plots.return_value = []

            create_plots_from_input(temp_folder_with_csv_files, output_file="ignored.png")

            # Should print warning about ignored output_file
            warning_calls = [call for call in mock_echo.call_args_list if "Warning" in str(call)]
            assert len(warning_calls) > 0
