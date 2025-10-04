"""
Cross-Validation для временных рядов с защитой от data leakage

Этот модуль содержит функции для правильной кроссвалидации временных рядов,
учитывая что target_return_20d зависит от будущих цен.

Key features:
- Rolling window CV с gap между train и test
- Gap = 21 торговый день (защита от leakage для t+20 таргетов)
- Работа с торговыми днями (не календарными)
- Поддержка пропущенных дат (выходные, праздники)

Референс: docs/cross_validation.md
"""

from typing import Iterator
import datetime
import numpy as np
import pandas as pd


def get_trading_dates(df: pd.DataFrame) -> list[datetime.date]:
    """
    Возвращает отсортированный список уникальных торговых дат

    Args:
        df: DataFrame с колонкой 'begin' (datetime)

    Returns:
        Список дат в формате datetime.date, отсортированный по возрастанию

    Example:
        >>> dates = get_trading_dates(train_df)
        >>> len(dates)
        1217
        >>> dates[:3]
        [datetime.date(2020, 6, 19),
         datetime.date(2020, 6, 22),
         datetime.date(2020, 6, 23)]
    """
    if "begin" not in df.columns:
        raise ValueError("DataFrame must have 'begin' column")

    # Конвертируем в datetime если нужно
    if not pd.api.types.is_datetime64_any_dtype(df["begin"]):
        df = df.copy()
        df["begin"] = pd.to_datetime(df["begin"])

    # Получаем уникальные даты и сортируем
    trading_dates = sorted(df["begin"].dt.date.unique())

    return trading_dates


def compute_t_plus_n(
    df: pd.DataFrame, date: datetime.date, n: int = 20
) -> datetime.date:
    """
    Вычисляет t+N в торговых днях

    Args:
        df: DataFrame с торговыми данными (с колонкой 'begin')
        date: Начальная дата
        n: Количество торговых дней вперед (default: 20)

    Returns:
        Дата через N торговых дней

    Raises:
        ValueError: если date не найдена или t+n выходит за границы данных

    Example:
        >>> t_plus_20 = compute_t_plus_n(df, datetime.date(2024, 1, 15), n=20)
        >>> print(t_plus_20)
        2024-02-12  # Примерно 28 календарных дней

        >>> # Проверка что это действительно 20 торговых дней
        >>> dates = get_trading_dates(df)
        >>> idx = dates.index(datetime.date(2024, 1, 15))
        >>> dates[idx + 20]
        datetime.date(2024, 2, 12)
    """
    trading_dates = get_trading_dates(df)

    try:
        current_idx = trading_dates.index(date)
    except ValueError:
        raise ValueError(
            f"Date {date} not found in trading dates. "
            f"Available range: {trading_dates[0]} to {trading_dates[-1]}"
        )

    target_idx = current_idx + n

    if target_idx >= len(trading_dates):
        raise ValueError(
            f"t+{n} from {date} goes beyond available data. "
            f"Last available date: {trading_dates[-1]} "
            f"(would need date at index {target_idx}, have {len(trading_dates)} dates)"
        )

    return trading_dates[target_idx]


def rolling_window_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = 60,
    gap: int = 21,
    min_train_size: int = 200,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Rolling window cross-validation с gap для временных рядов

    Создает фолды с гарантией отсутствия data leakage:
    - Train и Test разделены gap торговых дней
    - gap = 21 гарантирует что target_return_20d не пересекается

    Схема (для n_splits=3, test_size=60, gap=21):
        Fold 1: [═══════ Train ═══════][Gap 21][Test 60]
        Fold 2: [═════ Train ═════][Gap 21][Test 60]
        Fold 3: [═══ Train ═══][Gap 21][Test 60]

    Args:
        df: DataFrame с торговыми данными (должен иметь колонку 'begin')
        n_splits: Количество фолдов (default: 5)
        test_size: Размер test в торговых днях (default: 60)
        gap: Зазор между train и test в торговых днях (default: 21)
        min_train_size: Минимальный размер train в торговых днях (default: 200)

    Yields:
        (train_df, test_df) для каждого фолда

    Raises:
        ValueError: если недостаточно данных для n_splits фолдов

    Example:
        >>> from finam.cv import rolling_window_cv
        >>> import pandas as pd
        >>>
        >>> train_df = pd.read_parquet('data/preprocessed/train.parquet')
        >>>
        >>> for fold_idx, (train, test) in enumerate(rolling_window_cv(train_df, n_splits=5)):
        ...     print(f"Fold {fold_idx + 1}:")
        ...     print(f"  Train: {len(train):4d} rows ({train['begin'].min().date()} to {train['begin'].max().date()})")
        ...     print(f"  Test:  {len(test):4d} rows ({test['begin'].min().date()} to {test['begin'].max().date()})")
        ...
        ...     # Обучить модель
        ...     model.fit(train)
        ...     metrics = model.evaluate(test)
        ...     print(f"  MAE 1d: {metrics['mae_1d']:.4f}")

    Note:
        - DataFrame должен быть отсортирован по 'begin'
        - Gap=21 выбран для защиты от leakage при t+20 таргетах
        - Если gap=21, последний train день имеет target_20d из дня train_end+20,
          который не пересекается с test_start
    """
    # Проверка наличия колонки begin
    if "begin" not in df.columns:
        raise ValueError("DataFrame must have 'begin' column")

    # Конвертируем begin в datetime если нужно
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["begin"]):
        df["begin"] = pd.to_datetime(df["begin"])

    # Получаем торговые даты
    trading_dates = get_trading_dates(df)
    total_days = len(trading_dates)

    # Проверка что достаточно данных
    required_days = n_splits * test_size + (n_splits - 1) * gap + min_train_size
    if total_days < required_days:
        raise ValueError(
            f"Not enough data for {n_splits} folds. "
            f"Required: {required_days} trading days, "
            f"Available: {total_days} trading days. "
            f"Try reducing n_splits or test_size."
        )

    # Генерируем фолды
    for fold_idx in range(n_splits):
        # Индексы в списке trading_dates
        test_end_idx = total_days - 1 - fold_idx * test_size
        test_start_idx = test_end_idx - test_size + 1

        train_end_idx = test_start_idx - gap - 1
        train_start_idx = 0

        # Проверка что train не слишком маленький
        train_size = train_end_idx - train_start_idx + 1
        if train_size < min_train_size:
            # Останавливаемся если train слишком маленький
            break

        # Даты для фильтрации
        train_start_date = trading_dates[train_start_idx]
        train_end_date = trading_dates[train_end_idx]
        test_start_date = trading_dates[test_start_idx]
        test_end_date = trading_dates[test_end_idx]

        # Создаем train и test DataFrames
        train_mask = (df["begin"].dt.date >= train_start_date) & (
            df["begin"].dt.date <= train_end_date
        )
        test_mask = (df["begin"].dt.date >= test_start_date) & (
            df["begin"].dt.date <= test_end_date
        )

        train_fold = df[train_mask].copy()
        test_fold = df[test_mask].copy()

        yield train_fold, test_fold


def evaluate_with_cv(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
    verbose: bool = True,
    **cv_kwargs,
) -> dict[str, list[float]]:
    """
    Оценка модели с помощью кроссвалидации

    Для каждого фолда:
    1. Обучить модель на train
    2. Предсказать на test
    3. Вычислить метрики
    4. Сохранить результаты

    Args:
        model: Объект модели с методами fit() и predict()
               fit(train_df, feature_cols) -> None
               predict(test_df) -> dict с pred_return_1d, pred_return_20d, etc.
               evaluate(test_df) -> dict с метриками
        df: DataFrame с данными (должен иметь feature_cols и target_*)
        feature_cols: Список названий признаков
        n_splits: Количество фолдов (default: 5)
        verbose: Печатать прогресс (default: True)
        **cv_kwargs: Дополнительные аргументы для rolling_window_cv()
                     (test_size, gap, min_train_size)

    Returns:
        Dict с метриками для каждого фолда:
        {
            'mae_1d': [fold1, fold2, fold3, fold4, fold5],
            'mae_20d': [...],
            'brier_1d': [...],
            'brier_20d': [...],
            'da_1d': [...],
            'da_20d': [...],
            'score_1d': [...],     # Если модель поддерживает
            'score_20d': [...],    # Если модель поддерживает
            'score_total': [...]   # Если модель поддерживает
        }

    Example:
        >>> from finam.model import LightGBMModel
        >>> from finam.cv import evaluate_with_cv
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # 1. Загрузить данные
        >>> train_df = pd.read_parquet('data/preprocessed/train.parquet')
        >>>
        >>> # 2. Получить feature columns
        >>> feature_cols = [col for col in train_df.columns
        ...                 if col not in ['ticker', 'begin', 'open', 'high', 'low',
        ...                                'close', 'volume', 'adj_close']
        ...                 and not col.startswith('target_')]
        >>>
        >>> # 3. Создать модель
        >>> model = LightGBMModel(n_estimators=300, learning_rate=0.05, max_depth=7)
        >>>
        >>> # 4. Оценить с CV
        >>> cv_results = evaluate_with_cv(
        ...     model=model,
        ...     df=train_df,
        ...     feature_cols=feature_cols,
        ...     n_splits=5,
        ...     test_size=60,
        ...     gap=21,
        ...     verbose=True
        ... )
        >>>
        >>> # 5. Анализ результатов
        >>> print(f"Mean MAE 1d:  {np.mean(cv_results['mae_1d']):.4f} ± {np.std(cv_results['mae_1d']):.4f}")
        >>> print(f"Mean MAE 20d: {np.mean(cv_results['mae_20d']):.4f} ± {np.std(cv_results['mae_20d']):.4f}")
        >>>
        >>> # 6. Проверка стабильности
        >>> stability = np.std(cv_results['mae_1d']) / np.mean(cv_results['mae_1d'])
        >>> if stability > 0.3:
        ...     print("⚠️ WARNING: Model is unstable (std > 30% of mean)")
    """
    # Инициализация результатов
    results = {
        "mae_1d": [],
        "mae_20d": [],
        "brier_1d": [],
        "brier_20d": [],
        "da_1d": [],
        "da_20d": [],
    }

    # Опциональные метрики (если модель поддерживает normalized scores)
    optional_metrics = ["score_1d", "score_20d", "score_total"]

    if verbose:
        print("=" * 80)
        print(f"CROSS-VALIDATION ({n_splits} folds)")
        print("=" * 80)

    # Проходим по каждому фолду
    for fold_idx, (train_fold, test_fold) in enumerate(
        rolling_window_cv(df, n_splits=n_splits, **cv_kwargs)
    ):
        if verbose:
            print(f"\nFold {fold_idx + 1}/{n_splits}:")
            print(
                f"  Train: {len(train_fold):5d} rows ({train_fold['begin'].min().date()} to {train_fold['begin'].max().date()})"
            )
            print(
                f"  Test:  {len(test_fold):5d} rows ({test_fold['begin'].min().date()} to {test_fold['begin'].max().date()})"
            )

        # Обучение
        model.fit(train_fold, feature_cols)

        # Оценка
        metrics = model.evaluate(test_fold)

        # Сохранение результатов
        for metric_name in [
            "mae_1d",
            "mae_20d",
            "brier_1d",
            "brier_20d",
            "da_1d",
            "da_20d",
        ]:
            results[metric_name].append(metrics[metric_name])

        # Опциональные метрики
        for metric_name in optional_metrics:
            if metric_name in metrics:
                if metric_name not in results:
                    results[metric_name] = []
                results[metric_name].append(metrics[metric_name])

        if verbose:
            print(f"  MAE 1d:  {metrics['mae_1d']:.6f}")
            print(f"  MAE 20d: {metrics['mae_20d']:.6f}")
            print(f"  Brier 1d:  {metrics['brier_1d']:.6f}")
            print(f"  Brier 20d: {metrics['brier_20d']:.6f}")
            print(f"  DA 1d:  {metrics['da_1d']:.4f} ({metrics['da_1d'] * 100:.2f}%)")
            print(f"  DA 20d: {metrics['da_20d']:.4f} ({metrics['da_20d'] * 100:.2f}%)")

            # Показать score если доступен
            if "score_total" in metrics:
                print(f"  Score total: {metrics['score_total']:.6f}")

    if verbose:
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 80)

        for metric_name in results.keys():
            values = results[metric_name]
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)

            print(
                f"{metric_name:15s}: {mean_val:.6f} ± {std_val:.6f}  [{min_val:.6f}, {max_val:.6f}]"
            )

        # Проверка стабильности
        mae_1d_stability = np.std(results["mae_1d"]) / np.mean(results["mae_1d"])
        if mae_1d_stability > 0.3:
            print("\n⚠️ WARNING: Model appears unstable (std > 30% of mean for MAE 1d)")
            print(f"   Stability ratio: {mae_1d_stability:.2%}")
            print("   Consider: reduce model complexity, add regularization")

        print("=" * 80)

    return results


def purged_group_time_series_split(
    df: pd.DataFrame, n_splits: int = 5, test_size: int = 60, embargo: int = 21
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Advanced: Purged Group Time Series Split

    Альтернативная стратегия CV с учетом:
    - Группировка по тикерам (каждый тикер — независимая серия)
    - Embargo period (аналог gap, но применяется после train)
    - Purging overlapping samples

    Args:
        df: DataFrame с торговыми данными
        n_splits: Количество фолдов
        test_size: Размер test в торговых днях
        embargo: Период embargo после train (аналог gap)

    Yields:
        (train_indices, test_indices) для каждого фолда

    Note:
        Это более сложная версия для продвинутых экспериментов.
        Для базового использования рекомендуется rolling_window_cv().
        Референс: Lopez de Prado "Advances in Financial Machine Learning"
    """
    # TODO: Реализовать purged split с группировкой по тикерам
    # Сейчас используем rolling_window_cv как основную стратегию
    raise NotImplementedError(
        "purged_group_time_series_split not yet implemented. "
        "Use rolling_window_cv() for now."
    )
