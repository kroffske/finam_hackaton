"""
Collect Experiments Metrics

Собирает метрики из всех экспериментов в outputs/ и создает сводную таблицу.
Автоматически вычисляет normalized scores относительно baseline.

Usage:
    python scripts/collect_experiments.py
    python scripts/collect_experiments.py --output experiments_summary.csv
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

# Добавляем src/ в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np

from finam.metrics import normalized_score


def load_baseline_metrics(split: str = 'test') -> dict:
    """
    Загрузить baseline метрики из data/baseline_metrics.json

    Args:
        split: 'train', 'val', или 'test'

    Returns:
        dict с baseline метриками
    """
    baseline_path = project_root / 'data' / 'baseline_metrics.json'

    if not baseline_path.exists():
        print(f"WARNING: Baseline metrics not found at {baseline_path}")
        print("   Run: python scripts/compute_baseline_metrics.py")
        return None

    with open(baseline_path, 'r') as f:
        baseline_metrics = json.load(f)

    return baseline_metrics.get(split, None)


def collect_experiment_metrics(exp_dir: Path, baseline_metrics: dict = None) -> dict:
    """
    Собрать метрики из одного эксперимента

    Args:
        exp_dir: путь к директории эксперимента
        baseline_metrics: baseline метрики для нормализации (опционально)

    Returns:
        dict с метриками эксперимента
    """
    # Загружаем config
    config_path = exp_dir / 'config.yaml'
    if not config_path.exists():
        return None

    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Загружаем metrics.json
    metrics_path = exp_dir / 'metrics.json'
    if not metrics_path.exists():
        return None

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Извлекаем основную информацию
    exp_name = config['experiment']['name']
    model_type = config['experiment']['model_type']
    timestamp = config['experiment']['timestamp']
    run_id = exp_dir.name

    # Собираем метрики для всех splits
    result = {
        'run_id': run_id,
        'exp_name': exp_name,
        'model_type': model_type,
        'timestamp': timestamp,
    }

    # Для каждого split добавляем метрики
    for split in ['train', 'val', 'test']:
        if split not in metrics:
            continue

        split_metrics = metrics[split]
        prefix = f'{split}_'

        # Базовые метрики
        result[f'{prefix}mae_1d'] = split_metrics.get('mae_1d')
        result[f'{prefix}mae_20d'] = split_metrics.get('mae_20d')
        result[f'{prefix}brier_1d'] = split_metrics.get('brier_1d')
        result[f'{prefix}brier_20d'] = split_metrics.get('brier_20d')
        result[f'{prefix}da_1d'] = split_metrics.get('da_1d')
        result[f'{prefix}da_20d'] = split_metrics.get('da_20d')

        # Вычисляем normalized scores если есть baseline
        if baseline_metrics and split in baseline_metrics:
            baseline = baseline_metrics[split]

            # Score 1d
            score_1d = normalized_score(
                split_metrics['mae_1d'],
                split_metrics['brier_1d'],
                split_metrics['da_1d'],
                baseline['mae_1d'],
                baseline['brier_1d']
            )
            result[f'{prefix}score_1d'] = score_1d

            # Score 20d
            score_20d = normalized_score(
                split_metrics['mae_20d'],
                split_metrics['brier_20d'],
                split_metrics['da_20d'],
                baseline['mae_20d'],
                baseline['brier_20d']
            )
            result[f'{prefix}score_20d'] = score_20d

            # Total score (среднее)
            result[f'{prefix}score_total'] = (score_1d + score_20d) / 2

    return result


def main(output_path: str = None, split: str = 'test'):
    """
    Собрать метрики из всех экспериментов

    Args:
        output_path: путь для сохранения CSV (по умолчанию outputs/experiments_log.csv)
        split: какой split использовать для сортировки ('train', 'val', 'test')
    """
    print("=" * 80)
    print("COLLECT EXPERIMENTS METRICS")
    print("=" * 80 + "\n")

    if output_path is None:
        output_path = project_root / 'outputs' / 'experiments_log.csv'
    else:
        output_path = Path(output_path)

    # ========================================================================
    # 1. Загрузка baseline метрик
    # ========================================================================
    print(f"[1/3] Loading baseline metrics (split={split})...")

    baseline_metrics_all = {}
    baseline_path = project_root / 'data' / 'baseline_metrics.json'

    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_metrics_all = json.load(f)
        print(f"   OK Loaded baseline metrics from {baseline_path}")
    else:
        print(f"   WARNING: No baseline metrics found")
        print(f"   Run: python scripts/compute_baseline_metrics.py")

    # ========================================================================
    # 2. Сбор метрик из всех экспериментов
    # ========================================================================
    print(f"\n[2/3] Collecting metrics from outputs/...")

    outputs_dir = project_root / 'outputs'

    if not outputs_dir.exists():
        print(f"   ERROR: No outputs directory found: {outputs_dir}")
        return

    experiment_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]

    if not experiment_dirs:
        print(f"   WARNING: No experiments found in {outputs_dir}")
        return

    print(f"   Found {len(experiment_dirs)} experiment directories")

    experiments = []

    for exp_dir in sorted(experiment_dirs):
        exp_data = collect_experiment_metrics(exp_dir, baseline_metrics_all)

        if exp_data:
            experiments.append(exp_data)
            print(f"      - {exp_dir.name}: {exp_data.get('exp_name', 'unknown')}")

    if not experiments:
        print(f"   ERROR: No valid experiments with metrics.json found")
        return

    # ========================================================================
    # 3. Создание сводной таблицы
    # ========================================================================
    print(f"\n[3/3] Creating summary table...")

    df = pd.DataFrame(experiments)

    # Сортируем по test_score_total (лучшие сверху)
    score_col = f'{split}_score_total'
    if score_col in df.columns:
        df = df.sort_values(score_col, ascending=False)

    # Сохраняем
    df.to_csv(output_path, index=False)

    print(f"   OK Saved to: {output_path}")

    # ========================================================================
    # Вывод топ экспериментов
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"TOP EXPERIMENTS (by {split}_score_total)")
    print("=" * 80 + "\n")

    # Колонки для отображения
    display_cols = [
        'run_id', 'exp_name', 'model_type',
        f'{split}_mae_1d', f'{split}_mae_20d',
        f'{split}_brier_1d', f'{split}_brier_20d',
        f'{split}_da_1d', f'{split}_da_20d',
        f'{split}_score_1d', f'{split}_score_20d', f'{split}_score_total'
    ]

    # Фильтруем только существующие колонки
    display_cols = [c for c in display_cols if c in df.columns]

    # Показываем топ-10
    print(df[display_cols].head(10).to_string(index=False))

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE!")
    print("=" * 80 + "\n")

    print(f"Summary:")
    print(f"   Total experiments: {len(df)}")

    if score_col in df.columns:
        best_exp = df.iloc[0]
        print(f"   Best experiment: {best_exp['exp_name']} ({best_exp['run_id']})")
        print(f"   Best score: {best_exp[score_col]:.6f}")

    print(f"\n   Saved to: {output_path}")

    print("\nNext steps:")
    if score_col in df.columns and len(df) > 0:
        best_run_id = df.iloc[0]['run_id']
        print(f"   # Generate submission for best model")
        print(f"   python scripts/4_generate_submission.py --run-id {best_run_id}")
        print()

    print("   # View all experiments in Python")
    print(f"   df = pd.read_csv('outputs/experiments_log.csv')")
    print(f"   df.sort_values('{split}_score_total', ascending=False)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect metrics from all experiments')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: outputs/experiments_log.csv)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to use for sorting (default: test)')

    args = parser.parse_args()

    main(output_path=args.output, split=args.split)
