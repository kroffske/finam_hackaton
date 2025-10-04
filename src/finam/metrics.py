"""
Метрики для оценки качества прогнозов

Метрика соревнования:
- MAE (Mean Absolute Error) — точность прогноза доходности
"""

import numpy as np


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


# Удалены функции brier_score, directional_accuracy, normalized_score
# так как новая постановка задачи требует только минимизации MAE


def evaluate_predictions(
    y_true_dict: dict,
    pred_dict: dict,
    horizons: list[int] = None
) -> dict:
    """
    Оценка метрик для всех 20 горизонтов (только MAE)

    Args:
        y_true_dict: dict с реальными доходностями {
            'target_return_1d': array,
            'target_return_2d': array,
            ...
        }
        pred_dict: dict с предсказаниями {
            'pred_return_1d': array,
            'pred_return_2d': array,
            ...
        }
        horizons: список горизонтов для оценки (по умолчанию 1-20)

    Returns:
        dict с метриками:
            - mae_1d, mae_2d, ..., mae_20d: MAE для каждого горизонта
            - mae_mean: среднее MAE по всем горизонтам
    """
    if horizons is None:
        horizons = list(range(1, 21))

    results = {}
    mae_values = []

    # MAE для каждого горизонта
    for horizon in horizons:
        target_key = f'target_return_{horizon}d'
        pred_key = f'pred_return_{horizon}d'

        if target_key in y_true_dict and pred_key in pred_dict:
            mae_value = mae(y_true_dict[target_key], pred_dict[pred_key])
            results[f'mae_{horizon}d'] = mae_value
            if not np.isnan(mae_value):
                mae_values.append(mae_value)

    # Среднее MAE по всем горизонтам
    if mae_values:
        results['mae_mean'] = np.mean(mae_values)
    else:
        results['mae_mean'] = np.nan

    return results


def print_metrics(metrics: dict, model_name: str = "Model", show_all: bool = False):
    """
    Красивый вывод метрик для всех 20 горизонтов

    Args:
        metrics: dict с метриками (результат evaluate_predictions)
        model_name: название модели
        show_all: показать все 20 горизонтов или только ключевые (1, 5, 10, 15, 20 + mean)
    """
    print(f"\n{'='*70}")
    print(f"[METRICS] {model_name}")
    print(f"{'='*70}\n")

    if show_all:
        # Показываем все 20 горизонтов
        for horizon in range(1, 21):
            key = f'mae_{horizon}d'
            if key in metrics:
                print(f"  MAE {horizon:2d}d: {metrics[key]:.6f}")
    else:
        # Показываем только ключевые горизонты
        key_horizons = [1, 5, 10, 15, 20]
        for horizon in key_horizons:
            key = f'mae_{horizon}d'
            if key in metrics:
                print(f"  MAE {horizon:2d}d: {metrics[key]:.6f}")

    # Mean MAE
    print(f"\n{'='*70}")
    print(f"  MEAN MAE (all horizons): {metrics.get('mae_mean', np.nan):.6f}")
    print(f"{'='*70}\n")
