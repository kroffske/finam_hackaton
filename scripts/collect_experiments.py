"""
Collect Experiments Metrics

Собирает метрики из всех экспериментов в outputs/ и создает сводную таблицу.
Анализирует важность признаков для выбора лучших комбинаций.

Features:
- Собирает MAE для всех 20 горизонтов (1d-20d) + mae_mean
- Показывает ключевые горизонты (1, 5, 10, 15, 20) или все 20
- Анализ feature importance (топ-N признаков)
- Статистика встречаемости признаков по всем экспериментам

Usage:
    # Базовый сбор (ключевые горизонты)
    python scripts/collect_experiments.py

    # Все 20 горизонтов
    python scripts/collect_experiments.py --show-all-horizons

    # С анализом признаков
    python scripts/collect_experiments.py --analyze-features --top-n 15

    # Полный анализ
    python scripts/collect_experiments.py --analyze-features --show-all-horizons --split test
"""

import sys
from pathlib import Path
import argparse
import json

# Добавляем src/ в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd


def load_feature_importance(exp_dir: Path) -> pd.DataFrame:
    """
    Загрузить feature importance из эксперимента

    Args:
        exp_dir: путь к директории эксперимента

    Returns:
        DataFrame с важностью признаков или None
    """
    fi_path = exp_dir / 'feature_importance.csv'

    if not fi_path.exists():
        return None

    df = pd.read_csv(fi_path)
    return df


def collect_experiment_metrics(exp_dir: Path, include_features: bool = False, top_n: int = 10) -> dict:
    """
    Собрать метрики из одного эксперимента

    Args:
        exp_dir: путь к директории эксперимента
        include_features: включить анализ feature importance
        top_n: количество топ признаков для сохранения

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

    # Количество признаков
    n_features = config.get('features', {}).get('count', 0)

    # Собираем метрики для всех splits
    result = {
        'run_id': run_id,
        'exp_name': exp_name,
        'model_type': model_type,
        'timestamp': timestamp,
        'n_features': n_features,
    }

    # Для каждого split добавляем MAE метрики для всех 20 горизонтов
    for split in ['train', 'val', 'test']:
        if split not in metrics:
            continue

        split_metrics = metrics[split]
        prefix = f'{split}_'

        # MAE метрики для всех 20 горизонтов
        for horizon in range(1, 21):
            key = f'mae_{horizon}d'
            result[f'{prefix}{key}'] = split_metrics.get(key)

        # MAE mean
        result[f'{prefix}mae_mean'] = split_metrics.get('mae_mean')

    # Анализ feature importance
    if include_features:
        fi_df = load_feature_importance(exp_dir)
        if fi_df is not None and 'importance_mean' in fi_df.columns:
            # Сортируем по средней важности
            fi_df = fi_df.sort_values('importance_mean', ascending=False)

            # Топ признаки
            top_features = fi_df['feature'].head(top_n).tolist()
            result['top_features'] = ','.join(top_features)

            # Средняя важность топ-3
            if len(fi_df) >= 3:
                result['top3_importance_mean'] = fi_df['importance_mean'].head(3).mean()

    return result


def main(output_path: str = None, split: str = 'val', analyze_features: bool = False, top_n: int = 10, show_all_horizons: bool = False):
    """
    Собрать метрики из всех экспериментов

    Args:
        output_path: путь для сохранения CSV (по умолчанию outputs/experiments_log.csv)
        split: какой split использовать для сортировки ('train', 'val', 'test')
        analyze_features: включить детальный анализ feature importance
        top_n: количество топ признаков для отображения
        show_all_horizons: показать все 20 горизонтов в таблице
    """
    print("=" * 80)
    print("COLLECT EXPERIMENTS METRICS")
    print("=" * 80 + "\n")

    if output_path is None:
        output_path = project_root / 'outputs' / 'experiments_log.csv'
    else:
        output_path = Path(output_path)

    # ========================================================================
    # 1. Сбор метрик из всех экспериментов
    # ========================================================================
    print("[1/2] Collecting metrics from outputs/...")

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
        exp_data = collect_experiment_metrics(exp_dir, include_features=analyze_features, top_n=top_n)

        if exp_data:
            experiments.append(exp_data)
            print(f"      - {exp_dir.name}: {exp_data.get('exp_name', 'unknown')} ({exp_data.get('n_features', 0)} features)")

    if not experiments:
        print("   ERROR: No valid experiments with metrics.json found")
        return

    # ========================================================================
    # 2. Создание сводной таблицы
    # ========================================================================
    print("\n[2/2] Creating summary table...")

    df = pd.DataFrame(experiments)

    # Сортируем по mae_mean (меньше = лучше)
    mae_col = f'{split}_mae_mean'
    if mae_col in df.columns:
        df = df.sort_values(mae_col, ascending=True)

    # Сохраняем
    df.to_csv(output_path, index=False)

    print(f"   OK Saved to: {output_path}")

    # ========================================================================
    # Вывод топ экспериментов
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"TOP EXPERIMENTS (by {split}_mae_mean, lower is better)")
    print("=" * 80 + "\n")

    # Колонки для отображения
    if show_all_horizons:
        # Показываем все 20 горизонтов
        display_cols = ['exp_name', 'n_features']
        for h in range(1, 21):
            display_cols.append(f'{split}_mae_{h}d')
        display_cols.append(f'{split}_mae_mean')
    else:
        # Показываем только ключевые горизонты (1d, 5d, 10d, 15d, 20d, mean)
        display_cols = [
            'exp_name', 'n_features',
            f'{split}_mae_1d',
            f'{split}_mae_5d',
            f'{split}_mae_10d',
            f'{split}_mae_15d',
            f'{split}_mae_20d',
            f'{split}_mae_mean'
        ]

    # Добавляем top3_importance_mean если есть
    if analyze_features and 'top3_importance_mean' in df.columns:
        display_cols.append('top3_importance_mean')

    # Фильтруем только существующие колонки
    display_cols = [c for c in display_cols if c in df.columns]

    # Показываем топ-10
    print(df[display_cols].head(10).to_string(index=False))

    # ========================================================================
    # Feature Importance Analysis (если включен)
    # ========================================================================
    if analyze_features and 'top_features' in df.columns:
        print("\n" + "=" * 80)
        print(f"TOP {top_n} FEATURES (best experiment)")
        print("=" * 80 + "\n")

        best_exp = df.iloc[0]
        best_run_id = best_exp['run_id']

        # Загружаем feature importance для лучшего эксперимента
        best_exp_dir = outputs_dir / best_run_id
        fi_df = load_feature_importance(best_exp_dir)

        if fi_df is not None:
            fi_df = fi_df.sort_values('importance_mean', ascending=False)
            print(fi_df[['feature', 'importance_mean']].head(top_n).to_string(index=False))

            # Статистика по всем экспериментам
            print("\n" + "=" * 80)
            print("FEATURE IMPORTANCE STATISTICS")
            print("=" * 80 + "\n")

            # Подсчет встречаемости признаков в топ-N
            all_top_features = []
            for _, row in df.iterrows():
                if 'top_features' in row and pd.notna(row['top_features']):
                    all_top_features.extend(row['top_features'].split(','))

            if all_top_features:
                from collections import Counter
                feature_counts = Counter(all_top_features)
                print(f"Most frequent features in top-{top_n} across all experiments:")
                for feat, count in feature_counts.most_common(15):
                    print(f"   {feat:40s}: {count:2d} times")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE!")
    print("=" * 80 + "\n")

    print("Summary:")
    print(f"   Total experiments: {len(df)}")

    if mae_col in df.columns:
        best_exp = df.iloc[0]
        print(f"   Best experiment: {best_exp['exp_name']} ({best_exp['run_id']})")
        print(f"   Best MAE mean: {best_exp[mae_col]:.6f}")
        print(f"   Features count: {best_exp['n_features']}")

    print(f"\n   Saved to: {output_path}")

    print("\nNext steps:")
    if mae_col in df.columns and len(df) > 0:
        best_run_id = df.iloc[0]['run_id']
        print("   # Generate submission for best model")
        print(f"   python scripts/4_generate_submission.py --run-id {best_run_id}")
        print()

    print("   # View all experiments in Python")
    print("   df = pd.read_csv('outputs/experiments_log.csv')")
    print(f"   df.sort_values('{split}_mae_mean', ascending=True).head(10)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Collect metrics from all experiments and analyze feature importance'
    )
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: outputs/experiments_log.csv)')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Which split to use for sorting (default: val)')
    parser.add_argument('--analyze-features', action='store_true',
                        help='Enable detailed feature importance analysis')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top features to display (default: 10)')
    parser.add_argument('--show-all-horizons', action='store_true',
                        help='Show all 20 horizons in the table (default: show key horizons only)')

    args = parser.parse_args()

    main(
        output_path=args.output,
        split=args.split,
        analyze_features=args.analyze_features,
        top_n=args.top_n,
        show_all_horizons=args.show_all_horizons
    )
