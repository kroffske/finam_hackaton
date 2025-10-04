"""
Функции для сравнения моделей

Позволяет обучить несколько моделей и сравнить их метрики
"""

import pandas as pd
from typing import List, Dict

from .metrics import evaluate_predictions, print_metrics
from .model import BaseModel


def compare_models(
    models: Dict[str, BaseModel],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Сравнить несколько моделей

    Args:
        models: dict {model_name: model_instance}
        train_df: DataFrame с train данными (с таргетами)
        test_df: DataFrame с test данными (с таргетами для валидации)
        feature_columns: список колонок с features

    Returns:
        DataFrame с результатами сравнения

    Example:
        >>> models = {
        ...     'Momentum Baseline': MomentumBaseline(window_size=5),
        ...     'LightGBM': LightGBMModel(n_estimators=500)
        ... }
        >>> results = compare_models(models, train_df, test_df, feature_cols)
        >>> print(results)
    """
    print("="*80)
    print("[COMPARE] MODEL COMPARISON")
    print("="*80 + "\n")

    # Подготовка данных
    X_train = train_df[feature_columns].fillna(0)
    y_return_1d_train = train_df['target_return_1d'].values
    y_return_20d_train = train_df['target_return_20d'].values
    y_dir_1d_train = train_df['target_direction_1d'].values
    y_dir_20d_train = train_df['target_direction_20d'].values

    X_test = test_df[feature_columns].fillna(0)
    y_return_1d_test = test_df['target_return_1d'].values
    y_return_20d_test = test_df['target_return_20d'].values

    results_list = []

    # Сначала получим baseline метрики (для нормализации)
    baseline_metrics = None

    for model_name, model in models.items():
        print(f"\n{'='*80}")
        print(f"[MODEL] {model_name}")
        print(f"{'='*80}\n")

        # 1. Обучение
        print(f"[TRAIN] Training {model_name}...")
        model.fit(
            X_train,
            y_return_1d_train,
            y_return_20d_train,
            y_dir_1d_train,
            y_dir_20d_train
        )

        # 2. Предсказание на test
        print("[PREDICT] Predicting on test...")
        preds = model.predict(X_test)

        # 3. Оценка метрик
        if baseline_metrics is None:
            # Первая модель = baseline
            metrics = evaluate_predictions(
                y_return_1d_test,
                y_return_20d_test,
                preds['pred_return_1d'],
                preds['pred_return_20d'],
                preds['pred_prob_up_1d'],
                preds['pred_prob_up_20d']
            )
            baseline_metrics = metrics
        else:
            # Сравниваем с baseline
            metrics = evaluate_predictions(
                y_return_1d_test,
                y_return_20d_test,
                preds['pred_return_1d'],
                preds['pred_return_20d'],
                preds['pred_prob_up_1d'],
                preds['pred_prob_up_20d'],
                baseline_mae_1d=baseline_metrics['mae_1d'],
                baseline_mae_20d=baseline_metrics['mae_20d'],
                baseline_brier_1d=baseline_metrics['brier_1d'],
                baseline_brier_20d=baseline_metrics['brier_20d']
            )

        # Вывод метрик
        print_metrics(metrics, model_name=model_name)

        # Сохраняем результаты
        result_row = {
            'Model': model_name,
            'MAE_1d': metrics['mae_1d'],
            'MAE_20d': metrics['mae_20d'],
            'Brier_1d': metrics['brier_1d'],
            'Brier_20d': metrics['brier_20d'],
            'DA_1d': metrics['da_1d'],
            'DA_20d': metrics['da_20d']
        }

        if 'score_total' in metrics:
            result_row['Score_Total'] = metrics['score_total']
            result_row['Score_1d'] = metrics['score_1d']
            result_row['Score_20d'] = metrics['score_20d']

        results_list.append(result_row)

    # Создаём сводную таблицу
    results_df = pd.DataFrame(results_list)

    return results_df


def print_comparison_table(results_df: pd.DataFrame):
    """
    Красивый вывод таблицы сравнения

    Args:
        results_df: результат compare_models()
    """
    print("\n" + "="*80)
    print("[RESULTS] COMPARISON TABLE")
    print("="*80 + "\n")

    # Форматирование для красивого вывода
    pd.options.display.float_format = '{:.6f}'.format

    print(results_df.to_string(index=False))

    print("\n" + "="*80)

    # Выделяем лучшие результаты
    print("\n[BEST] BEST RESULTS:\n")

    # MAE (меньше = лучше)
    best_mae_1d = results_df.loc[results_df['MAE_1d'].idxmin()]
    print(f"  • Лучшая MAE (1d):   {best_mae_1d['Model']} ({best_mae_1d['MAE_1d']:.6f})")

    best_mae_20d = results_df.loc[results_df['MAE_20d'].idxmin()]
    print(f"  • Лучшая MAE (20d):  {best_mae_20d['Model']} ({best_mae_20d['MAE_20d']:.6f})")

    # Brier (меньше = лучше)
    best_brier_1d = results_df.loc[results_df['Brier_1d'].idxmin()]
    print(f"  • Лучший Brier (1d): {best_brier_1d['Model']} ({best_brier_1d['Brier_1d']:.6f})")

    best_brier_20d = results_df.loc[results_df['Brier_20d'].idxmin()]
    print(f"  • Лучший Brier (20d): {best_brier_20d['Model']} ({best_brier_20d['Brier_20d']:.6f})")

    # DA (больше = лучше)
    best_da_1d = results_df.loc[results_df['DA_1d'].idxmax()]
    print(f"  • Лучшая DA (1d):    {best_da_1d['Model']} ({best_da_1d['DA_1d']:.4f} = {best_da_1d['DA_1d']*100:.2f}%)")

    best_da_20d = results_df.loc[results_df['DA_20d'].idxmax()]
    print(f"  • Лучшая DA (20d):   {best_da_20d['Model']} ({best_da_20d['DA_20d']:.4f} = {best_da_20d['DA_20d']*100:.2f}%)")

    # Total Score (если есть)
    if 'Score_Total' in results_df.columns:
        best_score = results_df.loc[results_df['Score_Total'].idxmax()]
        print(f"\n  [WINNER] BEST MODEL (Total Score): {best_score['Model']} ({best_score['Score_Total']:.6f})")

    print("\n" + "="*80 + "\n")

    # Вычисляем улучшение относительно baseline
    if len(results_df) > 1:
        baseline = results_df.iloc[0]
        best_model = results_df.iloc[1:].loc[results_df.iloc[1:]['MAE_1d'].idxmin()] if len(results_df) > 1 else None

        if best_model is not None:
            print("[IMPROVEMENT] vs BASELINE:\n")

            mae_1d_improvement = (baseline['MAE_1d'] - best_model['MAE_1d']) / baseline['MAE_1d'] * 100
            mae_20d_improvement = (baseline['MAE_20d'] - best_model['MAE_20d']) / baseline['MAE_20d'] * 100

            print(f"  • MAE 1d:  {mae_1d_improvement:+.2f}% ({baseline['MAE_1d']:.6f} -> {best_model['MAE_1d']:.6f})")
            print(f"  • MAE 20d: {mae_20d_improvement:+.2f}% ({baseline['MAE_20d']:.6f} -> {best_model['MAE_20d']:.6f})")

            brier_1d_improvement = (baseline['Brier_1d'] - best_model['Brier_1d']) / baseline['Brier_1d'] * 100
            brier_20d_improvement = (baseline['Brier_20d'] - best_model['Brier_20d']) / baseline['Brier_20d'] * 100

            print(f"  • Brier 1d:  {brier_1d_improvement:+.2f}% ({baseline['Brier_1d']:.6f} -> {best_model['Brier_1d']:.6f})")
            print(f"  • Brier 20d: {brier_20d_improvement:+.2f}% ({baseline['Brier_20d']:.6f} -> {best_model['Brier_20d']:.6f})")

            da_1d_improvement = (best_model['DA_1d'] - baseline['DA_1d']) / baseline['DA_1d'] * 100
            da_20d_improvement = (best_model['DA_20d'] - baseline['DA_20d']) / baseline['DA_20d'] * 100

            print(f"  • DA 1d:  {da_1d_improvement:+.2f}% ({baseline['DA_1d']:.4f} -> {best_model['DA_1d']:.4f})")
            print(f"  • DA 20d: {da_20d_improvement:+.2f}% ({baseline['DA_20d']:.4f} -> {best_model['DA_20d']:.4f})")

            print("\n" + "="*80 + "\n")
