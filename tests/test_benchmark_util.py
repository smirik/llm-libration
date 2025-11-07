"""Tests for benchmark utility functions."""

import tempfile
import json
import csv
from pathlib import Path
from datetime import datetime
import pytest

from llm_libration.benchmark.util import (
    find_png_files,
    map_folder_to_expected_result,
    calculate_metrics,
    calculate_relaxed_metrics,
    save_benchmark_results,
)
from llm_libration.types import ResonanceType


class TestBenchmarkUtilities:
    """Test cases for benchmark utility functions."""

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

    def test_find_png_files(self, benchmark_directory):
        """Test find_png_files function."""
        png_files = find_png_files(benchmark_directory)

        # Should find 6 PNG files
        assert len(png_files) == 6

        # All should be PNG files
        for png_file in png_files:
            assert png_file.suffix == '.png'
            assert png_file.is_file()

        # Should be sorted
        assert png_files == sorted(png_files)

    def test_find_png_files_empty_directory(self):
        """Test find_png_files with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            png_files = find_png_files(Path(temp_dir))
            assert len(png_files) == 0

    def test_find_png_files_nested_structure(self):
        """Test find_png_files with deeply nested directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create nested structure
            nested_dir = temp_path / "level1" / "level2" / "level3"
            nested_dir.mkdir(parents=True)

            # Create PNG files at different levels
            png_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x18\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82'

            (temp_path / "root.png").write_bytes(png_content)
            (temp_path / "level1" / "level1.png").write_bytes(png_content)
            (nested_dir / "level3.png").write_bytes(png_content)

            png_files = find_png_files(temp_path)
            assert len(png_files) == 3

            # Check that all files are found
            filenames = [f.name for f in png_files]
            assert "root.png" in filenames
            assert "level1.png" in filenames
            assert "level3.png" in filenames

    def test_map_folder_to_expected_result(self):
        """Test folder name to expected result mapping."""
        # Test libration mapping
        assert map_folder_to_expected_result('libration') == ResonanceType.RESONANT
        assert map_folder_to_expected_result('LIBRATION') == ResonanceType.RESONANT
        assert map_folder_to_expected_result('slow_libration') == ResonanceType.RESONANT

        # Test circulation mapping
        assert map_folder_to_expected_result('circulation') == ResonanceType.NON_RESONANT
        assert map_folder_to_expected_result('CIRCULATION') == ResonanceType.NON_RESONANT
        assert map_folder_to_expected_result('slow_circulation') == ResonanceType.NON_RESONANT

        # Test non-resonant mapping
        assert map_folder_to_expected_result('non-resonant') == ResonanceType.NON_RESONANT
        assert map_folder_to_expected_result('NON-RESONANT') == ResonanceType.NON_RESONANT

        # Test transient mapping
        assert map_folder_to_expected_result('transient') == ResonanceType.TRANSIENT
        assert map_folder_to_expected_result('TRANSIENT') == ResonanceType.TRANSIENT

        # Test controversial mapping
        assert map_folder_to_expected_result('controversial') == ResonanceType.CONTROVERSIAL
        assert map_folder_to_expected_result('CONTROVERSIAL') == ResonanceType.CONTROVERSIAL

        # Test unknown mapping (defaults to controversial)
        assert map_folder_to_expected_result('unknown') == ResonanceType.CONTROVERSIAL

    def test_map_folder_to_expected_result_partial_matches(self):
        """Test folder mapping with partial matches."""
        # Test that partial matches work
        assert map_folder_to_expected_result('my_libration_data') == ResonanceType.RESONANT
        assert map_folder_to_expected_result('circulation_plots') == ResonanceType.NON_RESONANT
        assert map_folder_to_expected_result('transient_behavior') == ResonanceType.TRANSIENT

    def test_calculate_metrics_perfect_classification(self):
        """Test metrics calculation with perfect classification."""
        results = [
            {'expected_result': 'resonant', 'actual_result': 'resonant'},
            {'expected_result': 'resonant', 'actual_result': 'resonant'},
            {'expected_result': 'non-resonant', 'actual_result': 'non-resonant'},
            {'expected_result': 'non-resonant', 'actual_result': 'non-resonant'},
            {'expected_result': 'transient', 'actual_result': 'transient'},
            {'expected_result': 'controversial', 'actual_result': 'controversial'},
        ]

        metrics = calculate_metrics(results)

        assert metrics['total_files'] == 6
        assert metrics['true_positives'] == 4  # 2 resonant + 1 transient + 1 controversial
        assert metrics['true_negatives'] == 2  # 2 non-resonant
        assert metrics['false_positives'] == 0
        assert metrics['false_negatives'] == 0
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0

    def test_calculate_metrics_mixed_classification(self):
        """Test metrics calculation with mixed classification results."""
        results = [
            {'expected_result': 'resonant', 'actual_result': 'resonant'},  # TP
            {'expected_result': 'resonant', 'actual_result': 'non-resonant'},  # FN
            {'expected_result': 'non-resonant', 'actual_result': 'non-resonant'},  # TN
            {'expected_result': 'non-resonant', 'actual_result': 'resonant'},  # FP
        ]

        metrics = calculate_metrics(results)

        assert metrics['total_files'] == 4
        assert metrics['true_positives'] == 1
        assert metrics['true_negatives'] == 1
        assert metrics['false_positives'] == 1
        assert metrics['false_negatives'] == 1
        assert metrics['accuracy'] == 0.5
        assert metrics['precision'] == 0.5
        assert metrics['recall'] == 0.5
        assert metrics['f1_score'] == 0.5

    def test_calculate_metrics_empty_results(self):
        """Test metrics calculation with empty results."""
        results = []

        metrics = calculate_metrics(results)

        assert metrics['total_files'] == 0
        assert metrics['true_positives'] == 0
        assert metrics['true_negatives'] == 0
        assert metrics['false_positives'] == 0
        assert metrics['false_negatives'] == 0
        assert metrics['accuracy'] == 0.0
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1_score'] == 0.0

    def test_calculate_metrics_controversial_cases(self):
        """Test metrics calculation with controversial classifications."""
        results = [
            {'expected_result': 'controversial', 'actual_result': 'controversial'},  # TP
            {'expected_result': 'controversial', 'actual_result': 'resonant'},  # FP (controversial predicted as resonant)
            {'expected_result': 'controversial', 'actual_result': 'non-resonant'},  # FN (controversial predicted as non-resonant)
            {'expected_result': 'resonant', 'actual_result': 'controversial'},  # FN (resonant predicted as controversial)
            {'expected_result': 'non-resonant', 'actual_result': 'controversial'},  # FP (non-resonant predicted as controversial)
        ]

        metrics = calculate_metrics(results)

        assert metrics['total_files'] == 5
        assert metrics['true_positives'] == 1  # 1 correct controversial
        assert metrics['true_negatives'] == 0  # no correct non-resonant
        assert metrics['false_positives'] == 2  # 1 controversial->resonant + 1 non-resonant->controversial
        assert metrics['false_negatives'] == 2  # 1 controversial->non-resonant + 1 resonant->controversial

    def test_calculate_metrics_transient_cases(self):
        """Test metrics calculation with transient classifications."""
        results = [
            {'expected_result': 'transient', 'actual_result': 'transient'},  # TP
            {'expected_result': 'transient', 'actual_result': 'resonant'},  # FP (transient predicted as resonant)
            {'expected_result': 'transient', 'actual_result': 'non-resonant'},  # FN (transient predicted as non-resonant)
            {'expected_result': 'resonant', 'actual_result': 'transient'},  # FN (resonant predicted as transient)
            {'expected_result': 'non-resonant', 'actual_result': 'transient'},  # FP (non-resonant predicted as transient)
        ]

        metrics = calculate_metrics(results)

        assert metrics['total_files'] == 5
        assert metrics['true_positives'] == 1  # 1 correct transient
        assert metrics['true_negatives'] == 0  # no correct non-resonant
        assert metrics['false_positives'] == 2  # 1 transient->resonant + 1 non-resonant->transient
        assert metrics['false_negatives'] == 2  # 1 transient->non-resonant + 1 resonant->transient

    def test_calculate_metrics_all_false_positives(self):
        """Test metrics calculation with all false positives (no true positives)."""
        results = [
            {'expected_result': 'non-resonant', 'actual_result': 'resonant'},  # FP
            {'expected_result': 'non-resonant', 'actual_result': 'resonant'},  # FP
        ]

        metrics = calculate_metrics(results)

        assert metrics['total_files'] == 2
        assert metrics['true_positives'] == 0
        assert metrics['true_negatives'] == 0
        assert metrics['false_positives'] == 2
        assert metrics['false_negatives'] == 0
        assert metrics['accuracy'] == 0.0
        assert metrics['precision'] == 0.0  # 0 / (0 + 2) = 0
        assert metrics['recall'] == 0.0  # 0 / (0 + 0) = 0 (no actual positives)
        assert metrics['f1_score'] == 0.0

    def test_calculate_relaxed_metrics_perfect_acceptance(self):
        """Test relaxed metrics with all predictions being acceptable."""
        results = [
            # Resonant accepts: resonant, controversial
            {'expected_result': 'resonant', 'actual_result': 'resonant'},  # Correct
            {'expected_result': 'resonant', 'actual_result': 'controversial'},  # Acceptable
            # Non-resonant accepts: non-resonant, controversial, transient
            {'expected_result': 'non-resonant', 'actual_result': 'non-resonant'},  # Correct
            {'expected_result': 'non-resonant', 'actual_result': 'controversial'},  # Acceptable
            {'expected_result': 'non-resonant', 'actual_result': 'transient'},  # Acceptable
            # Transient accepts: transient, resonant, controversial
            {'expected_result': 'transient', 'actual_result': 'transient'},  # Correct
            {'expected_result': 'transient', 'actual_result': 'resonant'},  # Acceptable
            {'expected_result': 'transient', 'actual_result': 'controversial'},  # Acceptable
            # Controversial accepts: any value
            {'expected_result': 'controversial', 'actual_result': 'controversial'},  # Correct
            {'expected_result': 'controversial', 'actual_result': 'resonant'},  # Acceptable
            {'expected_result': 'controversial', 'actual_result': 'non-resonant'},  # Acceptable
            {'expected_result': 'controversial', 'actual_result': 'transient'},  # Acceptable
        ]

        metrics = calculate_relaxed_metrics(results)

        assert metrics['total_files'] == 12
        assert metrics['correct_predictions'] == 12  # All should be acceptable
        assert metrics['incorrect_predictions'] == 0
        assert metrics['accuracy'] == 1.0

    def test_calculate_relaxed_metrics_mixed_results(self):
        """Test relaxed metrics with mix of acceptable and unacceptable predictions."""
        results = [
            # Resonant cases
            {'expected_result': 'resonant', 'actual_result': 'resonant'},  # Acceptable
            {'expected_result': 'resonant', 'actual_result': 'controversial'},  # Acceptable
            {'expected_result': 'resonant', 'actual_result': 'non-resonant'},  # NOT acceptable
            {'expected_result': 'resonant', 'actual_result': 'transient'},  # NOT acceptable
            # Non-resonant cases
            {'expected_result': 'non-resonant', 'actual_result': 'non-resonant'},  # Acceptable
            {'expected_result': 'non-resonant', 'actual_result': 'controversial'},  # Acceptable
            {'expected_result': 'non-resonant', 'actual_result': 'transient'},  # Acceptable
            {'expected_result': 'non-resonant', 'actual_result': 'resonant'},  # NOT acceptable
            # Transient cases
            {'expected_result': 'transient', 'actual_result': 'transient'},  # Acceptable
            {'expected_result': 'transient', 'actual_result': 'resonant'},  # Acceptable
            {'expected_result': 'transient', 'actual_result': 'controversial'},  # Acceptable
            {'expected_result': 'transient', 'actual_result': 'non-resonant'},  # NOT acceptable
            # Controversial cases (all should be acceptable)
            {'expected_result': 'controversial', 'actual_result': 'controversial'},  # Acceptable
            {'expected_result': 'controversial', 'actual_result': 'resonant'},  # Acceptable
        ]

        metrics = calculate_relaxed_metrics(results)

        assert metrics['total_files'] == 14
        # Acceptable: 2 + 3 + 3 + 2 = 10
        assert metrics['correct_predictions'] == 10
        # Not acceptable: 2 + 1 + 1 + 0 = 4
        assert metrics['incorrect_predictions'] == 4
        assert metrics['accuracy'] == 10 / 14  # ≈ 0.714

    def test_calculate_relaxed_metrics_resonant_only(self):
        """Test relaxed metrics focusing on resonant expectations."""
        results = [
            {'expected_result': 'resonant', 'actual_result': 'resonant'},  # Acceptable
            {'expected_result': 'resonant', 'actual_result': 'controversial'},  # Acceptable
            {'expected_result': 'resonant', 'actual_result': 'non-resonant'},  # NOT acceptable
            {'expected_result': 'resonant', 'actual_result': 'transient'},  # NOT acceptable
        ]

        metrics = calculate_relaxed_metrics(results)

        assert metrics['total_files'] == 4
        assert metrics['correct_predictions'] == 2
        assert metrics['incorrect_predictions'] == 2
        assert metrics['accuracy'] == 0.5

    def test_calculate_relaxed_metrics_non_resonant_only(self):
        """Test relaxed metrics focusing on non-resonant expectations."""
        results = [
            {'expected_result': 'non-resonant', 'actual_result': 'non-resonant'},  # Acceptable
            {'expected_result': 'non-resonant', 'actual_result': 'controversial'},  # Acceptable
            {'expected_result': 'non-resonant', 'actual_result': 'transient'},  # Acceptable
            {'expected_result': 'non-resonant', 'actual_result': 'resonant'},  # NOT acceptable
        ]

        metrics = calculate_relaxed_metrics(results)

        assert metrics['total_files'] == 4
        assert metrics['correct_predictions'] == 3
        assert metrics['incorrect_predictions'] == 1
        assert metrics['accuracy'] == 0.75

    def test_calculate_relaxed_metrics_transient_only(self):
        """Test relaxed metrics focusing on transient expectations."""
        results = [
            {'expected_result': 'transient', 'actual_result': 'transient'},  # Acceptable
            {'expected_result': 'transient', 'actual_result': 'resonant'},  # Acceptable
            {'expected_result': 'transient', 'actual_result': 'controversial'},  # Acceptable
            {'expected_result': 'transient', 'actual_result': 'non-resonant'},  # NOT acceptable
        ]

        metrics = calculate_relaxed_metrics(results)

        assert metrics['total_files'] == 4
        assert metrics['correct_predictions'] == 3
        assert metrics['incorrect_predictions'] == 1
        assert metrics['accuracy'] == 0.75

    def test_calculate_relaxed_metrics_controversial_only(self):
        """Test relaxed metrics focusing on controversial expectations (should accept all)."""
        results = [
            {'expected_result': 'controversial', 'actual_result': 'controversial'},  # Acceptable
            {'expected_result': 'controversial', 'actual_result': 'resonant'},  # Acceptable
            {'expected_result': 'controversial', 'actual_result': 'non-resonant'},  # Acceptable
            {'expected_result': 'controversial', 'actual_result': 'transient'},  # Acceptable
        ]

        metrics = calculate_relaxed_metrics(results)

        assert metrics['total_files'] == 4
        assert metrics['correct_predictions'] == 4  # All should be acceptable
        assert metrics['incorrect_predictions'] == 0
        assert metrics['accuracy'] == 1.0

    def test_calculate_relaxed_metrics_empty_results(self):
        """Test relaxed metrics calculation with empty results."""
        results = []

        metrics = calculate_relaxed_metrics(results)

        assert metrics['total_files'] == 0
        assert metrics['correct_predictions'] == 0
        assert metrics['incorrect_predictions'] == 0
        assert metrics['accuracy'] == 0.0

    def test_calculate_relaxed_metrics_vs_strict_comparison(self):
        """Test that relaxed metrics are more forgiving than strict metrics."""
        # Results where relaxed should perform better than strict
        results = [
            {'expected_result': 'resonant', 'actual_result': 'controversial'},  # Strict: wrong, Relaxed: acceptable
            {'expected_result': 'non-resonant', 'actual_result': 'transient'},  # Strict: wrong, Relaxed: acceptable
            {'expected_result': 'transient', 'actual_result': 'resonant'},  # Strict: wrong, Relaxed: acceptable
            {'expected_result': 'controversial', 'actual_result': 'non-resonant'},  # Both should accept this
        ]

        strict_metrics = calculate_metrics(results)
        relaxed_metrics = calculate_relaxed_metrics(results)

        # Relaxed should have higher accuracy than strict
        assert relaxed_metrics['accuracy'] > strict_metrics['accuracy']
        assert relaxed_metrics['accuracy'] == 1.0  # All should be acceptable in relaxed
        assert strict_metrics['accuracy'] < 1.0  # Some should be wrong in strict

    def test_save_benchmark_results(self, benchmark_directory):
        """Test saving benchmark results to files."""
        results = [
            {
                'filename': 'test1.png',
                'full_path': 'libration/test1.png',
                'expected_result': 'resonant',
                'actual_result': 'resonant',
                'subtype': 'clear libration',
            },
            {
                'filename': 'test2.png',
                'full_path': 'circulation/test2.png',
                'expected_result': 'non-resonant',
                'actual_result': 'non-resonant',
                'subtype': 'circulation',
            },
        ]

        metrics = {
            'total_files': 2,
            'true_positives': 1,
            'true_negatives': 1,
            'false_positives': 0,
            'false_negatives': 0,
            'accuracy': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0,
        }

        start_time = datetime(2024, 3, 15, 14, 30, 45)  # Fixed datetime for testing

        csv_path, json_path = save_benchmark_results(benchmark_directory, results, metrics, 'openai', 'custom-model', start_time)

        # Check that files were created
        assert csv_path.exists()
        assert json_path.exists()

        # Check filename format (should use human-readable date/time)
        expected_date_time = "2024-03-15_14-30-45"
        assert expected_date_time in csv_path.name
        assert expected_date_time in json_path.name
        assert csv_path.name == f"benchmark_results_openai_custom-model_{expected_date_time}.csv"
        assert json_path.name == f"benchmark_details_openai_custom-model_{expected_date_time}.json"

        # Check CSV content
        with open(csv_path, 'r') as f:
            csv_reader = csv.DictReader(f)
            csv_results = list(csv_reader)
            assert len(csv_results) == 2
            assert csv_results[0]['filename'] == 'test1.png'
            assert csv_results[0]['expected_result'] == 'resonant'
            assert csv_results[0]['actual_result'] == 'resonant'
            assert csv_results[0]['subtype'] == 'clear libration'

        # Check JSON content
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            assert json_data['benchmark_info']['provider'] == 'openai'
            assert json_data['benchmark_info']['model'] == 'custom-model'
            assert json_data['benchmark_info']['total_files'] == 2
            assert json_data['metrics']['accuracy'] == 1.0
            assert len(json_data['detailed_results']) == 2

    def test_save_benchmark_results_default_model(self, benchmark_directory):
        """Test saving benchmark results with default model name."""
        results = [
            {
                'filename': 'test.png',
                'full_path': 'test/test.png',
                'expected_result': 'resonant',
                'actual_result': 'resonant',
                'subtype': 'clear libration',
            },
        ]

        metrics = {
            'total_files': 1,
            'true_positives': 1,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'accuracy': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0,
        }

        start_time = datetime.now()

        csv_path, json_path = save_benchmark_results(benchmark_directory, results, metrics, 'anthropic', None, start_time)

        # Check that files were created with default model name
        assert "anthropic_default" in csv_path.name
        assert "anthropic_default" in json_path.name

        # Check JSON content has default model
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            assert json_data['benchmark_info']['model'] == 'default'

    def test_save_benchmark_results_model_with_slash(self, benchmark_directory):
        """Test saving benchmark results with model name containing slash."""
        results = [
            {
                'filename': 'test.png',
                'full_path': 'test/test.png',
                'expected_result': 'resonant',
                'actual_result': 'resonant',
                'subtype': 'clear libration',
            },
        ]

        metrics = {
            'total_files': 1,
            'true_positives': 1,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'accuracy': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0,
        }

        start_time = datetime(2025, 10, 30, 14, 26, 56)

        # Test with a model name containing "/"
        csv_path, json_path = save_benchmark_results(
            benchmark_directory, results, metrics, 'openrouter', 'google/gemini-2.5-pro', start_time
        )

        # Check that files were created with sanitized model name (slash replaced with underscore)
        assert csv_path.exists()
        assert json_path.exists()
        assert "google_gemini-2.5-pro" in csv_path.name
        assert "google_gemini-2.5-pro" in json_path.name
        assert csv_path.name == "benchmark_results_openrouter_google_gemini-2.5-pro_2025-10-30_14-26-56.csv"
        assert json_path.name == "benchmark_details_openrouter_google_gemini-2.5-pro_2025-10-30_14-26-56.json"

        # Check JSON content preserves original model name
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            assert json_data['benchmark_info']['model'] == 'google/gemini-2.5-pro'  # Original name preserved in JSON
