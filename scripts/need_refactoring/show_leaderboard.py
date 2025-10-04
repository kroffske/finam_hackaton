"""
Show Experiments Leaderboard

Отображает ранжированный список всех экспериментов по test_score_total

Usage:
    python scripts/show_leaderboard.py
    python scripts/show_leaderboard.py --split val
    python scripts/show_leaderboard.py --top 5
"""

import sys
from pathlib import Path
import argparse

# Добавляем src/ в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd


def show_leaderboard(split: str = 'test', top: int = None):
    """
    Показать leaderboard экспериментов

    Args:
        split: какой split использовать ('train', 'val', 'test')
        top: показать только топ N экспериментов (None = все)
    """
    print("=" * 80)
    print(f"EXPERIMENTS LEADERBOARD (by {split}_score_total)")
    print("=" * 80)
    print()

    # Загрузка данных
    log_path = project_root / 'outputs' / 'experiments_log.csv'

    if not log_path.exists():
        print(f"ERROR: Experiments log not found: {log_path}")
        print()
        print("Run first:")
        print("   python scripts/collect_experiments.py")
        return

    df = pd.read_csv(log_path)

    # Фильтруем эксперименты с метриками для выбранного split
    score_col = f'{split}_score_total'
    df = df.dropna(subset=[score_col])

    if len(df) == 0:
        print(f"No experiments with {split} metrics found!")
        return

    # Сортировка по score (больше = лучше)
    df = df.sort_values(score_col, ascending=False)

    # Ограничение топ N
    if top is not None:
        df = df.head(top)

    # Находим baseline
    baseline_score = 0.050504  # default Momentum baseline
    baseline_row = df[df['model_type'] == 'momentum']
    if len(baseline_row) > 0:
        baseline_score = baseline_row[score_col].iloc[0]

    print(f"Rule: HIGHER = BETTER (baseline={baseline_score:.6f})")
    print()

    # Выводим таблицу
    print(f"{'Rank':<5} {'Model':^25} {'Score':>10} {'Improvement':>12} {'Type':^10}")
    print("-" * 80)

    for rank, (idx, row) in enumerate(df.iterrows(), 1):
        improvement = ((row[score_col] / baseline_score) - 1) * 100
        marker = '***' if improvement > 100 else ' **' if improvement > 50 else '  *' if improvement > 0 else '   '

        print(f"{marker} {rank:<2} {row['exp_name']:^25} {row[score_col]:>10.6f} {improvement:>+11.1f}% {row['model_type']:^10}")

    print("=" * 80)

    # Best model summary
    if len(df) > 0:
        best = df.iloc[0]
        best_improvement = ((best[score_col] / baseline_score) - 1) * 100

        print()
        print(f"BEST MODEL: {best['exp_name']}")
        print(f"  {split}_score_total: {best[score_col]:.6f}")
        print(f"  Improvement vs baseline: {best_improvement:+.1f}%")
        print()
        print(f"  Details:")
        print(f"    MAE 1d:  {best[f'{split}_mae_1d']:.6f}")
        print(f"    MAE 20d: {best[f'{split}_mae_20d']:.6f}")
        print(f"    Brier 1d:  {best[f'{split}_brier_1d']:.6f}")
        print(f"    Brier 20d: {best[f'{split}_brier_20d']:.6f}")
        print(f"    DA 1d:  {best[f'{split}_da_1d']:.4f} ({best[f'{split}_da_1d']*100:.2f}%)")
        print(f"    DA 20d: {best[f'{split}_da_20d']:.4f} ({best[f'{split}_da_20d']*100:.2f}%)")
        print()
        print(f"  Generate submission:")
        print(f"    python scripts/4_generate_submission.py --run-id {best['run_id']}")

    print("=" * 80)

    # Legend
    print()
    print("Legend:")
    print("  *** = Excellent (>100% improvement)")
    print("   ** = Good     (50-100% improvement)")
    print("    * = Baseline level (0-50%)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show experiments leaderboard')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to rank by (default: test)')
    parser.add_argument('--top', type=int, default=None,
                        help='Show only top N experiments (default: all)')

    args = parser.parse_args()

    show_leaderboard(split=args.split, top=args.top)
