"""Utility functions for benchmark evaluation."""

import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from llm_libration.types import ResonanceType


def find_png_files(directory: Path) -> List[Path]:
    """
    Recursively find all PNG files in a directory.

    Args:
        directory: Path to the directory to search

    Returns:
        List of Path objects for all PNG files found
    """
    png_files = []
    for png_file in directory.rglob("*.png"):
        if png_file.is_file():
            png_files.append(png_file)
    return sorted(png_files)


def map_folder_to_expected_result(folder_name: str) -> ResonanceType:
    """
    Map folder names to expected resonance types.

    Args:
        folder_name: Name of the folder

    Returns:
        Expected ResonanceType
    """
    folder_mapping = {
        'libration': ResonanceType.RESONANT,
        'circulation': ResonanceType.NON_RESONANT,
        'non-resonant': ResonanceType.NON_RESONANT,
        'transient': ResonanceType.CONTROVERSIAL,  # For now, map to controversial
        'controversial': ResonanceType.CONTROVERSIAL,
    }

    folder_lower = folder_name.lower()
    for key, value in folder_mapping.items():
        if key in folder_lower:
            return value

    # Default fallback
    return ResonanceType.CONTROVERSIAL


def calculate_metrics(results: List[Dict]) -> Dict:
    """
    Calculate benchmark metrics from results.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with calculated metrics
    """
    if not results:
        return {
            'total_files': 0,
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
        }

    # Count results by category
    tp = tn = fp = fn = 0

    for result in results:
        expected = result['expected_result']
        actual = result['actual_result']

        if expected == 'resonant' and actual == 'resonant':
            tp += 1
        elif expected == 'non-resonant' and actual == 'non-resonant':
            tn += 1
        elif expected == 'controversial' and actual == 'controversial':
            # Controversial correct predictions count as true positives
            tp += 1
        elif expected == 'resonant' and actual != 'resonant':
            fn += 1
        elif expected == 'non-resonant' and actual != 'non-resonant':
            fp += 1
        elif expected == 'controversial' and actual != 'controversial':
            # Controversial mispredictions depend on the actual prediction
            if actual == 'resonant':
                fp += 1
            else:  # actual == 'non-resonant'
                fn += 1
        else:
            # Handle other cases
            if actual == 'resonant':
                fp += 1
            else:
                fn += 1

    total = len(results)

    # Calculate metrics
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'total_files': total,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
    }


def save_benchmark_results(
    benchmark_dir: Path, results: List[Dict], metrics: Dict, provider: str, model: Optional[str], start_time: datetime
) -> Tuple[Path, Path]:
    """
    Save benchmark results to CSV and JSON files.

    Args:
        benchmark_dir: Directory where benchmark was run
        results: List of result dictionaries
        metrics: Calculated metrics
        provider: LLM provider used
        model: Model name used (if any)
        start_time: When the benchmark started

    Returns:
        Tuple of (csv_path, json_path)
    """
    # Use human-readable date and time format
    date_time = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    model_name = model or "default"

    # Save CSV results
    csv_filename = f"benchmark_results_{provider}_{model_name}_{date_time}.csv"
    csv_path = benchmark_dir / csv_filename

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'full_path', 'expected_result', 'actual_result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    # Save detailed JSON results
    json_filename = f"benchmark_details_{provider}_{model_name}_{date_time}.json"
    json_path = benchmark_dir / json_filename

    detailed_results = {
        'benchmark_info': {'timestamp': start_time.isoformat(), 'provider': provider, 'model': model_name, 'total_files': len(results)},
        'metrics': metrics,
        'file_list': [result['full_path'] for result in results],
        'detailed_results': results,
    }

    with open(json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(detailed_results, jsonfile, indent=2, ensure_ascii=False)

    return csv_path, json_path
