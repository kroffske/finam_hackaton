"""
Метрики для оценки качества прогнозов

Метрики соревнования:
- MAE (Mean Absolute Error) — точность прогноза доходности
- Brier Score — калибровка вероятностей
- DA (Directional Accuracy) — точность прогноза направления
- Normalized Score — комбинированная метрика относительно baseline
"""

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error

    Args:
        y_true: реальные значения (доходности)
        y_pred: предсказанные значения

    Returns:
        MAE score (меньше = лучше)

    Example:
        >>> y_true = np.array([0.01, -0.02, 0.03])
        >>> y_pred = np.array([0.015, -0.01, 0.025])
        >>> mae(y_true, y_pred)
        0.00833...
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Удаляем NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return np.nan

    return np.mean(np.abs(y_true - y_pred))


def brier_score(y_true: np.ndarray, prob_up: np.ndarray) -> float:
    """
    Brier Score — метрика калибровки вероятностей

    Проверяет: если модель говорит "вероятность роста 80%",
    то действительно ли в 80% случаев был рост?

    Args:
        y_true: реальные значения доходности (положительные = рост)
        prob_up: предсказанные вероятности роста [0, 1]

    Returns:
        Brier score (меньше = лучше, range [0, 1])

    Example:
        >>> y_true = np.array([0.01, -0.02, 0.03])  # 2 роста, 1 падение
        >>> prob_up = np.array([0.6, 0.3, 0.7])
        >>> brier_score(y_true, prob_up)
        0.153...
    """
    y_true = np.asarray(y_true)
    prob_up = np.asarray(prob_up)

    # Удаляем NaN
    mask = ~(np.isnan(y_true) | np.isnan(prob_up))
    y_true = y_true[mask]
    prob_up = prob_up[mask]

    if len(y_true) == 0:
        return np.nan

    # Преобразуем доходности в бинарные метки (1 = рост, 0 = падение)
    y_binary = (y_true > 0).astype(float)

    # Clip probabilities to [0, 1] для безопасности
    prob_up = np.clip(prob_up, 0.0, 1.0)

    # Brier Score = среднее квадратов разности
    return np.mean((y_binary - prob_up) ** 2)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy — доля правильно угаданных направлений

    Проверяет: совпадает ли знак предсказания с реальностью?

    Args:
        y_true: реальные доходности
        y_pred: предсказанные доходности

    Returns:
        DA score [0, 1] (больше = лучше, 0.5 = как монетка)

    Example:
        >>> y_true = np.array([0.01, -0.02, 0.03, -0.01])
        >>> y_pred = np.array([0.015, -0.01, 0.005, 0.01])
        >>> directional_accuracy(y_true, y_pred)
        0.75  # 3 из 4 правильных направлений
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Удаляем NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return np.nan

    # Знаки совпадают?
    same_direction = (np.sign(y_true) == np.sign(y_pred))

    return np.mean(same_direction)


def normalized_score(
    mae_model: float,
    brier_model: float,
    da_model: float,
    mae_baseline: float,
    brier_baseline: float,
    da_baseline: float = 0.5,
) -> float:
    """
    Нормализованный комбинированный score

    Formula (from competition):
        Score = 0.7 × MAE_norm + 0.3 × Brier_norm + 0.1 × DA

    where:
        MAE_norm = 1 - (model_MAE / baseline_MAE)
        Brier_norm = 1 - (model_Brier / baseline_Brier)
        DA = directional_accuracy (без нормализации, т.к. уже в [0,1])

    Args:
        mae_model: MAE вашей модели
        brier_model: Brier вашей модели
        da_model: DA вашей модели
        mae_baseline: MAE baseline модели
        brier_baseline: Brier baseline модели
        da_baseline: DA baseline (по умолчанию 0.5 = random)

    Returns:
        Normalized score (больше = лучше)

    Example:
        >>> normalized_score(
        ...     mae_model=0.03, brier_model=0.2, da_model=0.6,
        ...     mae_baseline=0.04, brier_baseline=0.25
        ... )
        0.43  # модель лучше baseline
    """
    # MAE и Brier: меньше = лучше → нормализуем как (1 - model/baseline)
    mae_norm = 1.0 - (mae_model / mae_baseline) if mae_baseline > 0 else 0.0
    brier_norm = 1.0 - (brier_model / brier_baseline) if brier_baseline > 0 else 0.0

    # DA: больше = лучше → используем как есть
    da_score = da_model

    # Комбинированный score
    score = 0.7 * mae_norm + 0.3 * brier_norm + 0.1 * da_score

    return score


def evaluate_predictions(
    y_true_1d: np.ndarray,
    y_true_20d: np.ndarray,
    pred_return_1d: np.ndarray,
    pred_return_20d: np.ndarray,
    pred_prob_up_1d: np.ndarray,
    pred_prob_up_20d: np.ndarray,
    baseline_mae_1d: float = None,
    baseline_mae_20d: float = None,
    baseline_brier_1d: float = None,
    baseline_brier_20d: float = None,
) -> dict:
    """
    Полная оценка всех метрик для модели

    Args:
        y_true_1d: реальные доходности 1d
        y_true_20d: реальные доходности 20d
        pred_return_1d: предсказанные доходности 1d
        pred_return_20d: предсказанные доходности 20d
        pred_prob_up_1d: предсказанные вероятности роста 1d
        pred_prob_up_20d: предсказанные вероятности роста 20d
        baseline_mae_1d: MAE baseline для 1d (если None, не считаем normalized score)
        baseline_mae_20d: MAE baseline для 20d
        baseline_brier_1d: Brier baseline для 1d
        baseline_brier_20d: Brier baseline для 20d

    Returns:
        dict с метриками:
            - mae_1d, mae_20d
            - brier_1d, brier_20d
            - da_1d, da_20d
            - score_1d, score_20d (если baseline предоставлен)
            - score_total (среднее)
    """
    results = {}

    # MAE
    results['mae_1d'] = mae(y_true_1d, pred_return_1d)
    results['mae_20d'] = mae(y_true_20d, pred_return_20d)

    # Brier Score
    results['brier_1d'] = brier_score(y_true_1d, pred_prob_up_1d)
    results['brier_20d'] = brier_score(y_true_20d, pred_prob_up_20d)

    # Directional Accuracy
    results['da_1d'] = directional_accuracy(y_true_1d, pred_return_1d)
    results['da_20d'] = directional_accuracy(y_true_20d, pred_return_20d)

    # Normalized Score (если baseline предоставлен)
    if all(x is not None for x in [baseline_mae_1d, baseline_brier_1d]):
        results['score_1d'] = normalized_score(
            results['mae_1d'], results['brier_1d'], results['da_1d'],
            baseline_mae_1d, baseline_brier_1d
        )

    if all(x is not None for x in [baseline_mae_20d, baseline_brier_20d]):
        results['score_20d'] = normalized_score(
            results['mae_20d'], results['brier_20d'], results['da_20d'],
            baseline_mae_20d, baseline_brier_20d
        )

    # Total score (среднее по горизонтам)
    if 'score_1d' in results and 'score_20d' in results:
        results['score_total'] = (results['score_1d'] + results['score_20d']) / 2

    return results


def print_metrics(metrics: dict, model_name: str = "Model"):
    """
    Красивый вывод метрик

    Args:
        metrics: dict с метриками (результат evaluate_predictions)
        model_name: название модели
    """
    print(f"\n{'='*70}")
    print(f"[METRICS] {model_name}")
    print(f"{'='*70}")

    # 1-day metrics
    print(f"\n1-DAY METRICS:")
    print(f"  MAE:        {metrics.get('mae_1d', np.nan):.6f}")
    print(f"  Brier:      {metrics.get('brier_1d', np.nan):.6f}")
    print(f"  DA:         {metrics.get('da_1d', np.nan):.4f} ({metrics.get('da_1d', 0)*100:.2f}%)")
    if 'score_1d' in metrics:
        print(f"  Score:      {metrics['score_1d']:.6f}")

    # 20-day metrics
    print(f"\n20-DAY METRICS:")
    print(f"  MAE:        {metrics.get('mae_20d', np.nan):.6f}")
    print(f"  Brier:      {metrics.get('brier_20d', np.nan):.6f}")
    print(f"  DA:         {metrics.get('da_20d', np.nan):.4f} ({metrics.get('da_20d', 0)*100:.2f}%)")
    if 'score_20d' in metrics:
        print(f"  Score:      {metrics['score_20d']:.6f}")

    # Total
    if 'score_total' in metrics:
        print(f"\n{'='*70}")
        print(f"  TOTAL SCORE: {metrics['score_total']:.6f}")

    print(f"{'='*70}\n")
