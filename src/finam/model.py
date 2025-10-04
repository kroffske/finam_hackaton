"""
Model wrappers для FORECAST задачи

- BaseModel: абстрактный интерфейс
- MomentumBaseline: обёртка над scripts/baseline_solution.py
- LightGBMModel: ML модель на LightGBM

Фокус только на прогнозе доходности (MAE), без вероятностей
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Абстрактный интерфейс для моделей

    Все модели должны реализовать:
    - fit(X, y_returns_dict) - обучение на 20 горизонтах
    - predict(X) -> dict с 20 предсказаниями
    """

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y_returns_dict: dict
    ):
        """
        Обучение модели

        Args:
            X: DataFrame с features
            y_returns_dict: dict с таргетами {
                'target_return_1d': np.ndarray,
                'target_return_2d': np.ndarray,
                ...
                'target_return_20d': np.ndarray
            }
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> dict:
        """
        Предсказание

        Args:
            X: DataFrame с features

        Returns:
            dict с ключами:
                - pred_return_1d: np.ndarray
                - pred_return_2d: np.ndarray
                ...
                - pred_return_20d: np.ndarray
        """
        pass


class MomentumBaseline(BaseModel):
    """
    Baseline модель на основе momentum

    Логика: pred_return_Nd = momentum * scaling_factor_N
    Линейная интерполяция scaling factor от 1d до 20d

    Не требует обучения (rule-based модель)
    """

    def __init__(
        self,
        window_size: int = 5,
        scaling_1d: float = 0.3,
        scaling_20d: float = 1.0
    ):
        """
        Args:
            window_size: размер окна для momentum
            scaling_1d: коэффициент для pred_return_1d
            scaling_20d: коэффициент для pred_return_20d
        """
        self.window_size = window_size
        self.scaling_1d = scaling_1d
        self.scaling_20d = scaling_20d

    def fit(self, X, y_returns_dict):
        """Baseline не требует обучения (rule-based)"""
        pass

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Предсказание на основе momentum для всех 20 горизонтов

        Args:
            X: DataFrame, должен содержать колонку momentum_{window_size}d

        Returns:
            dict с предсказаниями доходности для 1-20 дней
        """
        # Извлекаем momentum
        momentum_col = f'momentum_{self.window_size}d'

        if momentum_col not in X.columns:
            raise ValueError(f"Column '{momentum_col}' not found in X. Run add_all_features first!")

        momentum = X[momentum_col].fillna(0).values

        # Предсказание доходности для всех горизонтов
        predictions = {}

        for horizon in range(1, 21):
            # Линейная интерполяция scaling factor
            scaling = self.scaling_1d + (self.scaling_20d - self.scaling_1d) * (horizon - 1) / 19

            pred_return = momentum * scaling
            # Clipping пропорционально горизонту
            max_return = 0.2 + (0.5 - 0.2) * (horizon - 1) / 19
            pred_return = np.clip(pred_return, -max_return, max_return)

            predictions[f'pred_return_{horizon}d'] = pred_return

        return predictions


class LightGBMModel(BaseModel):
    """
    LightGBM модель для прогнозирования доходности

    Обучает 20 регрессионных моделей:
    - return_1d, return_2d, ..., return_20d: regression (MAE loss)
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        verbose: int = -1
    ):
        """
        Args:
            n_estimators: количество деревьев
            learning_rate: learning rate
            max_depth: максимальная глубина дерева
            num_leaves: максимальное количество листьев
            min_child_samples: минимальное количество сэмплов в листе
            subsample: доля сэмплов для каждого дерева
            colsample_bytree: доля признаков для каждого дерева
            random_state: seed для воспроизводимости
            verbose: уровень логирования LightGBM
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM не установлен. Установите: pip install lightgbm"
            )

        self.lgb = lgb
        self.params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'verbose': verbose
        }

        # Модели для каждого горизонта (будут созданы в fit)
        self.models = {}  # {1: model_1d, 2: model_2d, ..., 20: model_20d}

        self.feature_names = None

    def fit(
        self,
        X: pd.DataFrame,
        y_returns_dict: dict
    ):
        """
        Обучение 20 регрессионных моделей

        Args:
            X: DataFrame с features
            y_returns_dict: dict с таргетами для всех горизонтов
                {'target_return_1d': array, ..., 'target_return_20d': array}
        """
        print("[AI] Training LightGBM models for 20 horizons...")

        # Сохраняем имена признаков
        self.feature_names = X.columns.tolist()

        # Обработка NaN
        X = X.fillna(0)

        # Обучаем модель для каждого горизонта
        for horizon in range(1, 21):
            target_key = f'target_return_{horizon}d'

            if target_key not in y_returns_dict:
                print(f"   WARNING: {target_key} not found, skipping...")
                continue

            y_target = y_returns_dict[target_key]

            # Удаляем строки с NaN в таргете
            mask = ~np.isnan(y_target)

            print(f"   • Training return_{horizon}d model ({mask.sum()} samples)...")

            model = self.lgb.LGBMRegressor(
                objective='mae',
                **self.params
            )
            model.fit(
                X[mask],
                y_target[mask]
            )

            self.models[horizon] = model

        print(f"   [OK] Training complete! Trained {len(self.models)} models.\n")

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Предсказание доходности для всех 20 горизонтов

        Args:
            X: DataFrame с features

        Returns:
            dict с предсказаниями доходности {
                'pred_return_1d': array,
                'pred_return_2d': array,
                ...
                'pred_return_20d': array
            }
        """
        if not self.models:
            raise ValueError("Model not trained yet. Call fit() first!")

        # Убедимся что колонки те же что при обучении
        if set(X.columns) != set(self.feature_names):
            missing = set(self.feature_names) - set(X.columns)
            extra = set(X.columns) - set(self.feature_names)
            if missing:
                print(f"[WARN] Missing features: {missing}")
            if extra:
                print(f"[WARN] Extra features (will be ignored): {extra}")

        # Приводим к нужным колонкам
        X = X[self.feature_names].fillna(0)

        # Предсказания для всех горизонтов
        predictions = {}

        for horizon in range(1, 21):
            if horizon not in self.models:
                print(f"[WARN] Model for horizon {horizon}d not found, skipping...")
                continue

            pred = self.models[horizon].predict(X)

            # Clipping пропорционально горизонту
            max_return = 0.5 + (1.0 - 0.5) * (horizon - 1) / 19
            pred = np.clip(pred, -max_return, max_return)

            predictions[f'pred_return_{horizon}d'] = pred

        return predictions

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Получить feature importance усредненную по всем горизонтам

        Args:
            importance_type: 'gain' или 'split'

        Returns:
            DataFrame с колонками [feature, importance_1d, importance_5d, ..., importance_20d, importance_mean]
        """
        if not self.models:
            raise ValueError("Model not trained yet")

        importance_df = pd.DataFrame({
            'feature': self.feature_names
        })

        # Importance для каждого горизонта
        importance_cols = []
        for horizon in sorted(self.models.keys()):
            col_name = f'importance_{horizon}d'
            importance_df[col_name] = self.models[horizon].feature_importances_
            importance_cols.append(col_name)

        # Средняя importance по всем горизонтам
        importance_df['importance_mean'] = importance_df[importance_cols].mean(axis=1)

        # Сортируем по средней importance
        importance_df = importance_df.sort_values('importance_mean', ascending=False)

        return importance_df


# CalibratedLightGBMModel удалён - больше не нужен
# Калибровка вероятностей не требуется при фокусе только на MAE
