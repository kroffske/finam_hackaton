"""
Визуализация работы Baseline решения

Создаёт графики:
1. Sigmoid с разными sensitivity
2. Linear vs Sigmoid
3. Calibration curve (reliability diagram)
4. Brier Score vs clipping strategies
5. Распределение вероятностей
"""

import sys
import io
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка matplotlib для красивых графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def sigmoid(x, sensitivity=10):
    """Sigmoid функция"""
    return 1 / (1 + np.exp(-sensitivity * x))


def compute_momentum(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Вычисление momentum для каждого тикера"""
    df = df.sort_values(['ticker', 'begin']).copy()

    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_data = df[mask].copy()

        # Momentum = процентное изменение за window дней
        ticker_data['momentum'] = ticker_data['close'].pct_change(window)

        df.loc[mask, 'momentum'] = ticker_data['momentum'].values

    return df


def brier_score(y_true: np.ndarray, prob_up: np.ndarray) -> float:
    """Brier Score для оценки калибровки"""
    y_binary = (y_true > 0).astype(float)
    prob_up = np.clip(prob_up, 0.0, 1.0)
    return np.mean((y_binary - prob_up) ** 2)


def plot_sigmoid_variants(output_dir: Path):
    """График 1: Sigmoid с разными sensitivity"""
    print("\n📊 График 1: Sigmoid с разными sensitivity...")

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.linspace(-0.2, 0.2, 1000)
    sensitivities = [5, 10, 20]

    for sens in sensitivities:
        y = sigmoid(x, sensitivity=sens)
        ax.plot(x, y, label=f'sensitivity={sens}', linewidth=2)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% (neutral)')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Momentum', fontsize=12)
    ax.set_ylabel('Probability of Up', fontsize=12)
    ax.set_title('Sigmoid Function with Different Sensitivity Parameters', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Аннотации
    ax.text(0.05, 0.62, 'High momentum\n→ High prob', fontsize=10, ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(-0.05, 0.38, 'Low momentum\n→ Low prob', fontsize=10, ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_sigmoid_variants.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Сохранено: {output_dir / 'baseline_sigmoid_variants.png'}")


def plot_linear_vs_sigmoid(output_dir: Path):
    """График 2: Linear (returns) vs Sigmoid (probabilities)"""
    print("\n📊 График 2: Linear vs Sigmoid...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    momentum = np.linspace(-0.15, 0.15, 1000)

    # Linear для returns
    pred_return_1d = momentum * 0.3
    pred_return_20d = momentum * 1.0

    ax1.plot(momentum, pred_return_1d, label='pred_return_1d (× 0.3)', linewidth=2)
    ax1.plot(momentum, pred_return_20d, label='pred_return_20d (× 1.0)', linewidth=2, linestyle='--')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Momentum', fontsize=12)
    ax1.set_ylabel('Predicted Return', fontsize=12)
    ax1.set_title('LINEAR: Returns Prediction', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Sigmoid для probabilities
    prob_up_1d = sigmoid(momentum, sensitivity=10)
    prob_up_20d = sigmoid(momentum, sensitivity=5)

    ax2.plot(momentum, prob_up_1d, label='pred_prob_up_1d (sens=10)', linewidth=2)
    ax2.plot(momentum, prob_up_20d, label='pred_prob_up_20d (sens=5)', linewidth=2, linestyle='--')
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% neutral')
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Momentum', fontsize=12)
    ax2.set_ylabel('Probability of Up', fontsize=12)
    ax2.set_title('SIGMOID: Probability Prediction', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_linear_vs_sigmoid.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Сохранено: {output_dir / 'baseline_linear_vs_sigmoid.png'}")


def plot_calibration_curve(y_true: np.ndarray, prob_pred: np.ndarray,
                           output_dir: Path, horizon: str = '1d'):
    """График 3: Calibration curve (reliability diagram)"""
    print(f"\n📊 График 3: Calibration curve ({horizon})...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Убираем NaN
    mask = ~(np.isnan(y_true) | np.isnan(prob_pred))
    y_true = y_true[mask]
    prob_pred = prob_pred[mask]

    # Бинарные метки
    y_binary = (y_true > 0).astype(float)

    # Разбиваем на бины
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    observed_freq = []
    predicted_freq = []
    counts = []

    for i in range(n_bins):
        mask = (prob_pred >= bins[i]) & (prob_pred < bins[i+1])
        if mask.sum() > 0:
            observed_freq.append(y_binary[mask].mean())
            predicted_freq.append(prob_pred[mask].mean())
            counts.append(mask.sum())
        else:
            observed_freq.append(np.nan)
            predicted_freq.append(bin_centers[i])
            counts.append(0)

    # График
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration', alpha=0.7)

    # Scatter с размером пропорциональным количеству
    sizes = np.array(counts) / max(counts) * 500
    scatter = ax.scatter(predicted_freq, observed_freq, s=sizes, alpha=0.6,
                        c=range(n_bins), cmap='viridis', edgecolors='black', linewidth=1)

    # Соединяем линией
    valid_mask = ~np.isnan(observed_freq)
    ax.plot(np.array(predicted_freq)[valid_mask], np.array(observed_freq)[valid_mask],
           'b-', linewidth=2, alpha=0.5, label='Baseline calibration')

    ax.set_xlabel('Predicted Probability', fontsize=14)
    ax.set_ylabel('Observed Frequency', fontsize=14)
    ax.set_title(f'Calibration Curve (Reliability Diagram) — {horizon}',
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Colorbar для бинов
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Probability Bin', fontsize=12)

    # Текст с метрикой
    brier = brier_score(y_true, prob_pred)
    ax.text(0.05, 0.95, f'Brier Score: {brier:.4f}',
           transform=ax.transAxes, fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / f'baseline_calibration_{horizon}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Сохранено: {output_dir / f'baseline_calibration_{horizon}.png'}")


def plot_brier_vs_clipping(df: pd.DataFrame, output_dir: Path):
    """График 4: Brier Score vs clipping strategies"""
    print("\n📊 График 4: Brier Score vs clipping...")

    # Вычисляем momentum
    df = compute_momentum(df, window=5)
    df = df.dropna(subset=['momentum', 'target_return_1d', 'target_return_20d'])

    # Базовые вероятности (без clipping)
    prob_base_1d = sigmoid(df['momentum'].values, sensitivity=10)
    prob_base_20d = sigmoid(df['momentum'].values, sensitivity=5)

    # Разные стратегии clipping
    clipping_strategies = [
        ('No clipping', (0.0, 1.0)),
        ('Clip [0.1, 0.9]', (0.1, 0.9)),
        ('Clip [0.2, 0.8]', (0.2, 0.8)),
        ('Clip [0.05, 0.95]', (0.05, 0.95)),
        ('Clip [0.15, 0.85]', (0.15, 0.85)),
    ]

    results_1d = []
    results_20d = []

    for strategy_name, (low, high) in clipping_strategies:
        prob_clipped_1d = np.clip(prob_base_1d, low, high)
        prob_clipped_20d = np.clip(prob_base_20d, low, high)

        brier_1d = brier_score(df['target_return_1d'].values, prob_clipped_1d)
        brier_20d = brier_score(df['target_return_20d'].values, prob_clipped_20d)

        results_1d.append({'Strategy': strategy_name, 'Brier': brier_1d})
        results_20d.append({'Strategy': strategy_name, 'Brier': brier_20d})

    # График
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_results_1d = pd.DataFrame(results_1d)
    df_results_20d = pd.DataFrame(results_20d)

    # 1d
    bars1 = ax1.barh(df_results_1d['Strategy'], df_results_1d['Brier'], color='skyblue', edgecolor='black')
    ax1.set_xlabel('Brier Score', fontsize=12)
    ax1.set_title('Brier Score vs Clipping Strategy (1d)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Выделяем лучшую
    best_idx_1d = df_results_1d['Brier'].idxmin()
    bars1[best_idx_1d].set_color('lightgreen')

    # Добавляем значения
    for i, (idx, row) in enumerate(df_results_1d.iterrows()):
        ax1.text(row['Brier'], i, f" {row['Brier']:.4f}",
                va='center', fontsize=10, fontweight='bold')

    # 20d
    bars2 = ax2.barh(df_results_20d['Strategy'], df_results_20d['Brier'], color='salmon', edgecolor='black')
    ax2.set_xlabel('Brier Score', fontsize=12)
    ax2.set_title('Brier Score vs Clipping Strategy (20d)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # Выделяем лучшую
    best_idx_20d = df_results_20d['Brier'].idxmin()
    bars2[best_idx_20d].set_color('lightgreen')

    # Добавляем значения
    for i, (idx, row) in enumerate(df_results_20d.iterrows()):
        ax2.text(row['Brier'], i, f" {row['Brier']:.4f}",
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_brier_vs_clipping.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Сохранено: {output_dir / 'baseline_brier_vs_clipping.png'}")

    # Выводим таблицу в консоль
    print("\n   📋 Результаты:")
    print("\n   1-DAY:")
    print(df_results_1d.to_string(index=False))
    print("\n   20-DAY:")
    print(df_results_20d.to_string(index=False))


def plot_probability_distribution(df: pd.DataFrame, output_dir: Path):
    """График 5: Распределение вероятностей с/без clipping"""
    print("\n📊 График 5: Распределение вероятностей...")

    # Вычисляем momentum
    df = compute_momentum(df, window=5)
    df = df.dropna(subset=['momentum'])

    # Базовые вероятности
    prob_base_1d = sigmoid(df['momentum'].values, sensitivity=10)
    prob_clipped_1d = np.clip(prob_base_1d, 0.1, 0.9)

    prob_base_20d = sigmoid(df['momentum'].values, sensitivity=5)
    prob_clipped_20d = np.clip(prob_base_20d, 0.1, 0.9)

    # График
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1d без clipping
    axes[0, 0].hist(prob_base_1d, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(prob_base_1d.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {prob_base_1d.mean():.3f}')
    axes[0, 0].set_xlabel('Probability', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('1d Probabilities (No Clipping)', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # 1d с clipping
    axes[0, 1].hist(prob_clipped_1d, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(prob_clipped_1d.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {prob_clipped_1d.mean():.3f}')
    axes[0, 1].axvline(0.1, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Clip bounds')
    axes[0, 1].axvline(0.9, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Probability', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('1d Probabilities (Clipped [0.1, 0.9])', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # 20d без clipping
    axes[1, 0].hist(prob_base_20d, bins=50, alpha=0.7, color='salmon', edgecolor='black')
    axes[1, 0].axvline(prob_base_20d.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {prob_base_20d.mean():.3f}')
    axes[1, 0].set_xlabel('Probability', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('20d Probabilities (No Clipping)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # 20d с clipping
    axes[1, 1].hist(prob_clipped_20d, bins=50, alpha=0.7, color='khaki', edgecolor='black')
    axes[1, 1].axvline(prob_clipped_20d.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {prob_clipped_20d.mean():.3f}')
    axes[1, 1].axvline(0.1, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Clip bounds')
    axes[1, 1].axvline(0.9, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Probability', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('20d Probabilities (Clipped [0.1, 0.9])', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_prob_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Сохранено: {output_dir / 'baseline_prob_distribution.png'}")


def main():
    print("=" * 80)
    print("📊 ВИЗУАЛИЗАЦИЯ BASELINE РЕШЕНИЯ")
    print("=" * 80)

    # Создаём директорию для графиков
    output_dir = Path(__file__).parent.parent / 'docs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Директория для графиков: {output_dir}")

    # Графики 1-2: теоретические (без данных)
    plot_sigmoid_variants(output_dir)
    plot_linear_vs_sigmoid(output_dir)

    # Загружаем данные для графиков 3-5
    print("\n📊 Загрузка данных...")
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'participants' / 'train_candles.csv'

    if not data_path.exists():
        print(f"\n⚠️  Файл не найден: {data_path}")
        print("   Графики 3-5 (требующие данных) будут пропущены.")
        print("   Сохранены только теоретические графики 1-2.")
        return

    df = pd.read_csv(data_path)
    df['begin'] = pd.to_datetime(df['begin'])
    print(f"   ✓ Загружено {len(df)} строк, {df['ticker'].nunique()} тикеров")

    # Используем последние 20% для валидации
    df = df.sort_values('begin')
    split_idx = int(len(df['begin'].unique()) * 0.8)
    split_date = sorted(df['begin'].unique())[split_idx]
    val_df = df[df['begin'] >= split_date].copy()

    print(f"   ✓ Validation: {len(val_df)} строк ({val_df['begin'].min()} - {val_df['begin'].max()})")

    # Вычисляем momentum
    val_df = compute_momentum(val_df, window=5)
    val_df = val_df.dropna(subset=['momentum', 'target_return_1d', 'target_return_20d'])

    print(f"   ✓ После удаления NaN: {len(val_df)} строк")

    # Вычисляем вероятности с clipping
    prob_1d = sigmoid(val_df['momentum'].values, sensitivity=10)
    prob_1d_clipped = np.clip(prob_1d, 0.1, 0.9)

    prob_20d = sigmoid(val_df['momentum'].values, sensitivity=5)
    prob_20d_clipped = np.clip(prob_20d, 0.1, 0.9)

    # Графики 3-5: на реальных данных
    plot_calibration_curve(val_df['target_return_1d'].values, prob_1d_clipped, output_dir, '1d')
    plot_calibration_curve(val_df['target_return_20d'].values, prob_20d_clipped, output_dir, '20d')
    plot_brier_vs_clipping(val_df, output_dir)
    plot_probability_distribution(val_df, output_dir)

    print("\n" + "=" * 80)
    print("✅ ВСЕ ГРАФИКИ СОЗДАНЫ!")
    print("=" * 80)
    print(f"\n📁 Графики сохранены в: {output_dir}")
    print("\n   Созданные файлы:")
    for file in sorted(output_dir.glob('baseline_*.png')):
        print(f"   - {file.name}")
    print("\n💡 Добавьте эти графики в docs/BASELINE.md")


if __name__ == "__main__":
    main()
