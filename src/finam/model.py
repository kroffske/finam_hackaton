"""
Model wrappers для FORECAST задачи

- BaseModel: абстрактный интерфейс
- MomentumBaseline: обёртка над scripts/baseline_solution.py
- LightGBMModel: ML модель на LightGBM
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Абстрактный интерфейс для моделей

    Все модели должны реализовать:
    - fit(X, y_return_1d, y_return_20d, y_dir_1d, y_dir_20d)
    - predict(X) -> dict
    """

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y_return_1d: np.ndarray,
        y_return_20d: np.ndarray,
        y_direction_1d: np.ndarray = None,
        y_direction_20d: np.ndarray = None
    ):
        """
        Обучение модели

        Args:
            X: DataFrame с features
            y_return_1d: таргет доходности 1d
            y_return_20d: таргет доходности 20d
            y_direction_1d: таргет направления 1d (опционально)
            y_direction_20d: таргет направления 20d (опционально)
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
                - pred_return_20d: np.ndarray
                - pred_prob_up_1d: np.ndarray
                - pred_prob_up_20d: np.ndarray
        """
        pass


class MomentumBaseline(BaseModel):
    """
    Baseline модель на основе momentum

    Логика (из scripts/baseline_solution.py):
    - pred_return = momentum * scaling_factor
    - pred_prob_up = sigmoid(momentum * sensitivity)

    Не требует обучения (rule-based модель)
    """

    def __init__(
        self,
        window_size: int = 5,
        scaling_1d: float = 0.3,
        scaling_20d: float = 1.0,
        sensitivity_1d: float = 10.0,
        sensitivity_20d: float = 5.0
    ):
        """
        Args:
            window_size: размер окна для momentum
            scaling_1d: коэффициент для pred_return_1d
            scaling_20d: коэффициент для pred_return_20d
            sensitivity_1d: чувствительность сигмоиды для prob_up_1d
            sensitivity_20d: чувствительность сигмоиды для prob_up_20d
        """
        self.window_size = window_size
        self.scaling_1d = scaling_1d
        self.scaling_20d = scaling_20d
        self.sensitivity_1d = sensitivity_1d
        self.sensitivity_20d = sensitivity_20d

    def fit(self, X, y_return_1d, y_return_20d, y_direction_1d=None, y_direction_20d=None):
        """Baseline не требует обучения (rule-based)"""
        pass

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Предсказание на основе momentum

        Args:
            X: DataFrame, должен содержать колонку momentum_{window_size}d

        Returns:
            dict с предсказаниями
        """
        # Извлекаем momentum
        momentum_col = f'momentum_{self.window_size}d'

        if momentum_col not in X.columns:
            raise ValueError(f"Column '{momentum_col}' not found in X. Run add_all_features first!")

        momentum = X[momentum_col].fillna(0).values

        # Предсказание доходности
        pred_return_1d = momentum * self.scaling_1d
        pred_return_20d = momentum * self.scaling_20d

        # Clipping доходностей
        pred_return_1d = np.clip(pred_return_1d, -0.2, 0.2)
        pred_return_20d = np.clip(pred_return_20d, -0.5, 0.5)

        # Предсказание вероятности роста (sigmoid)
        def sigmoid(x, sensitivity):
            return 1 / (1 + np.exp(-sensitivity * x))

        pred_prob_up_1d = sigmoid(momentum, self.sensitivity_1d)
        pred_prob_up_20d = sigmoid(momentum, self.sensitivity_20d)

        # Clipping вероятностей
        pred_prob_up_1d = np.clip(pred_prob_up_1d, 0.1, 0.9)
        pred_prob_up_20d = np.clip(pred_prob_up_20d, 0.1, 0.9)

        return {
            'pred_return_1d': pred_return_1d,
            'pred_return_20d': pred_return_20d,
            'pred_prob_up_1d': pred_prob_up_1d,
            'pred_prob_up_20d': pred_prob_up_20d
        }


class LightGBMModel(BaseModel):
    """
    LightGBM модель для прогнозирования

    Обучает 4 отдельных модели:
    - return_1d: regression (MAE loss)
    - return_20d: regression (MAE loss)
    - prob_up_1d: binary classification
    - prob_up_20d: binary classification
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

        # Модели (будут созданы в fit)
        self.model_return_1d = None
        self.model_return_20d = None
        self.model_prob_up_1d = None
        self.model_prob_up_20d = None

        self.feature_names = None

    def fit(
        self,
        X: pd.DataFrame,
        y_return_1d: np.ndarray,
        y_return_20d: np.ndarray,
        y_direction_1d: np.ndarray = None,
        y_direction_20d: np.ndarray = None
    ):
        """
        Обучение 4 моделей

        Args:
            X: DataFrame с features
            y_return_1d: таргет доходности 1d
            y_return_20d: таргет доходности 20d
            y_direction_1d: таргет направления 1d (0/1)
            y_direction_20d: таргет направления 20d (0/1)
        """
        print("[AI] Training LightGBM models...")

        # Сохраняем имена признаков
        self.feature_names = X.columns.tolist()

        # Обработка NaN
        X = X.fillna(0)

        # Удаляем строки с NaN в таргетах
        mask_1d = ~np.isnan(y_return_1d)
        mask_20d = ~np.isnan(y_return_20d)

        # 1. Return 1d (regression, MAE loss)
        print("   • Training return_1d model...")
        self.model_return_1d = self.lgb.LGBMRegressor(
            objective='mae',
            **self.params
        )
        self.model_return_1d.fit(
            X[mask_1d],
            y_return_1d[mask_1d]
        )

        # 2. Return 20d (regression, MAE loss)
        print("   • Training return_20d model...")
        self.model_return_20d = self.lgb.LGBMRegressor(
            objective='mae',
            **self.params
        )
        self.model_return_20d.fit(
            X[mask_20d],
            y_return_20d[mask_20d]
        )

        # 3. Probability up 1d (binary classification)
        if y_direction_1d is not None:
            print("   • Training prob_up_1d model...")
            mask_dir_1d = ~np.isnan(y_direction_1d)

            self.model_prob_up_1d = self.lgb.LGBMClassifier(
                objective='binary',
                **self.params
            )
            self.model_prob_up_1d.fit(
                X[mask_dir_1d],
                y_direction_1d[mask_dir_1d]
            )
        else:
            print("   [WARN] y_direction_1d not provided, using sigmoid(return_1d) for prob_up_1d")
            self.model_prob_up_1d = None

        # 4. Probability up 20d (binary classification)
        if y_direction_20d is not None:
            print("   • Training prob_up_20d model...")
            mask_dir_20d = ~np.isnan(y_direction_20d)

            self.model_prob_up_20d = self.lgb.LGBMClassifier(
                objective='binary',
                **self.params
            )
            self.model_prob_up_20d.fit(
                X[mask_dir_20d],
                y_direction_20d[mask_dir_20d]
            )
        else:
            print("   [WARN] y_direction_20d not provided, using sigmoid(return_20d) for prob_up_20d")
            self.model_prob_up_20d = None

        print("   [OK] Training complete!\n")

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Предсказание

        Args:
            X: DataFrame с features

        Returns:
            dict с предсказаниями
        """
        if self.model_return_1d is None:
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

        # Предсказания доходностей
        pred_return_1d = self.model_return_1d.predict(X)
        pred_return_20d = self.model_return_20d.predict(X)

        # Clipping доходностей
        pred_return_1d = np.clip(pred_return_1d, -0.5, 0.5)
        pred_return_20d = np.clip(pred_return_20d, -1.0, 1.0)

        # Предсказания вероятностей
        if self.model_prob_up_1d is not None:
            pred_prob_up_1d = self.model_prob_up_1d.predict_proba(X)[:, 1]
        else:
            # Fallback: sigmoid от предсказанной доходности
            pred_prob_up_1d = 1 / (1 + np.exp(-10 * pred_return_1d))

        if self.model_prob_up_20d is not None:
            pred_prob_up_20d = self.model_prob_up_20d.predict_proba(X)[:, 1]
        else:
            # Fallback: sigmoid от предсказанной доходности
            pred_prob_up_20d = 1 / (1 + np.exp(-5 * pred_return_20d))

        # Clipping вероятностей
        pred_prob_up_1d = np.clip(pred_prob_up_1d, 0.01, 0.99)
        pred_prob_up_20d = np.clip(pred_prob_up_20d, 0.01, 0.99)

        return {
            'pred_return_1d': pred_return_1d,
            'pred_return_20d': pred_return_20d,
            'pred_prob_up_1d': pred_prob_up_1d,
            'pred_prob_up_20d': pred_prob_up_20d
        }

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Получить feature importance

        Args:
            importance_type: 'gain' или 'split'

        Returns:
            DataFrame с колонками [feature, importance_return_1d, importance_return_20d, ...]
        """
        if self.model_return_1d is None:
            raise ValueError("Model not trained yet")

        importance_df = pd.DataFrame({
            'feature': self.feature_names
        })

        # Importance для каждой модели
        importance_df['importance_return_1d'] = self.model_return_1d.feature_importances_
        importance_df['importance_return_20d'] = self.model_return_20d.feature_importances_

        if self.model_prob_up_1d is not None:
            importance_df['importance_prob_up_1d'] = self.model_prob_up_1d.feature_importances_

        if self.model_prob_up_20d is not None:
            importance_df['importance_prob_up_20d'] = self.model_prob_up_20d.feature_importances_

        # Средняя importance
        importance_cols = [col for col in importance_df.columns if col.startswith('importance_')]
        importance_df['importance_mean'] = importance_df[importance_cols].mean(axis=1)

        # Сортируем по средней importance
        importance_df = importance_df.sort_values('importance_mean', ascending=False)

        return importance_df


class CalibratedLightGBMModel(BaseModel):
    """
    LightGBM модель с калибровкой вероятностей

    Использует CalibratedClassifierCV для улучшения Brier score.
    Обучает те же 4 модели что и LightGBMModel, но classification модели
    калибруются для лучшей оценки вероятностей.
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
        verbose: int = -1,
        calibration_method: str = 'isotonic',
        calibration_cv: int = 5
    ):
        """
        Args:
            ... (те же что у LightGBMModel)
            calibration_method: метод калибровки ('sigmoid' или 'isotonic')
            calibration_cv: количество фолдов для калибровки
        """
        try:
            import lightgbm as lgb
            from sklearn.calibration import CalibratedClassifierCV
        except ImportError:
            raise ImportError(
                "LightGBM или sklearn не установлены. Установите: pip install lightgbm scikit-learn"
            )

        self.lgb = lgb
        self.CalibratedClassifierCV = CalibratedClassifierCV

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

        self.calibration_method = calibration_method
        self.calibration_cv = calibration_cv

        # Модели (будут созданы в fit)
        self.model_return_1d = None
        self.model_return_20d = None
        self.model_prob_up_1d = None
        self.model_prob_up_20d = None

        self.feature_names = None

    def fit(
        self,
        X: pd.DataFrame,
        y_return_1d: np.ndarray,
        y_return_20d: np.ndarray,
        y_direction_1d: np.ndarray = None,
        y_direction_20d: np.ndarray = None
    ):
        """
        Обучение 4 моделей с калибровкой для classification

        Args:
            X: DataFrame с features
            y_return_1d: таргет доходности 1d
            y_return_20d: таргет доходности 20d
            y_direction_1d: таргет направления 1d (0/1)
            y_direction_20d: таргет направления 20d (0/1)
        """
        print("[AI] Training Calibrated LightGBM models...")

        # Сохраняем имена признаков
        self.feature_names = X.columns.tolist()

        # Обработка NaN
        X = X.fillna(0)

        # Удаляем строки с NaN в таргетах
        mask_1d = ~np.isnan(y_return_1d)
        mask_20d = ~np.isnan(y_return_20d)

        # 1. Return 1d (regression, MAE loss) — БЕЗ калибровки
        print("   • Training return_1d model (no calibration)...")
        self.model_return_1d = self.lgb.LGBMRegressor(
            objective='mae',
            **self.params
        )
        self.model_return_1d.fit(
            X[mask_1d],
            y_return_1d[mask_1d]
        )

        # 2. Return 20d (regression, MAE loss) — БЕЗ калибровки
        print("   • Training return_20d model (no calibration)...")
        self.model_return_20d = self.lgb.LGBMRegressor(
            objective='mae',
            **self.params
        )
        self.model_return_20d.fit(
            X[mask_20d],
            y_return_20d[mask_20d]
        )

        # 3. Probability up 1d (binary classification) — С КАЛИБРОВКОЙ
        if y_direction_1d is not None:
            print(f"   • Training prob_up_1d model WITH CALIBRATION ({self.calibration_method})...")
            mask_dir_1d = ~np.isnan(y_direction_1d)

            base_clf_1d = self.lgb.LGBMClassifier(
                objective='binary',
                **self.params
            )

            self.model_prob_up_1d = self.CalibratedClassifierCV(
                base_clf_1d,
                method=self.calibration_method,
                cv=self.calibration_cv
            )

            self.model_prob_up_1d.fit(
                X[mask_dir_1d],
                y_direction_1d[mask_dir_1d]
            )
        else:
            print("   [WARN] y_direction_1d not provided, using sigmoid(return_1d) for prob_up_1d")
            self.model_prob_up_1d = None

        # 4. Probability up 20d (binary classification) — С КАЛИБРОВКОЙ
        if y_direction_20d is not None:
            print(f"   • Training prob_up_20d model WITH CALIBRATION ({self.calibration_method})...")
            mask_dir_20d = ~np.isnan(y_direction_20d)

            base_clf_20d = self.lgb.LGBMClassifier(
                objective='binary',
                **self.params
            )

            self.model_prob_up_20d = self.CalibratedClassifierCV(
                base_clf_20d,
                method=self.calibration_method,
                cv=self.calibration_cv
            )

            self.model_prob_up_20d.fit(
                X[mask_dir_20d],
                y_direction_20d[mask_dir_20d]
            )
        else:
            print("   [WARN] y_direction_20d not provided, using sigmoid(return_20d) for prob_up_20d")
            self.model_prob_up_20d = None

        print("   [OK] Training with calibration complete!\n")

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Предсказание (с калибровкой для вероятностей)

        Args:
            X: DataFrame с features

        Returns:
            dict с предсказаниями
        """
        if self.model_return_1d is None:
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

        # Предсказания доходностей
        pred_return_1d = self.model_return_1d.predict(X)
        pred_return_20d = self.model_return_20d.predict(X)

        # Clipping доходностей
        pred_return_1d = np.clip(pred_return_1d, -0.5, 0.5)
        pred_return_20d = np.clip(pred_return_20d, -1.0, 1.0)

        # Предсказания вероятностей (КАЛИБРОВАННЫЕ!)
        if self.model_prob_up_1d is not None:
            pred_prob_up_1d = self.model_prob_up_1d.predict_proba(X)[:, 1]
        else:
            # Fallback: sigmoid от предсказанной доходности
            pred_prob_up_1d = 1 / (1 + np.exp(-10 * pred_return_1d))

        if self.model_prob_up_20d is not None:
            pred_prob_up_20d = self.model_prob_up_20d.predict_proba(X)[:, 1]
        else:
            # Fallback: sigmoid от предсказанной доходности
            pred_prob_up_20d = 1 / (1 + np.exp(-5 * pred_return_20d))

        # Clipping вероятностей
        pred_prob_up_1d = np.clip(pred_prob_up_1d, 0.01, 0.99)
        pred_prob_up_20d = np.clip(pred_prob_up_20d, 0.01, 0.99)

        return {
            'pred_return_1d': pred_return_1d,
            'pred_return_20d': pred_return_20d,
            'pred_prob_up_1d': pred_prob_up_1d,
            'pred_prob_up_20d': pred_prob_up_20d
        }

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Получить feature importance

        Note: Для калиброванных моделей берем importance из базовой модели
        (первый калибратор из CalibratedClassifierCV)

        Args:
            importance_type: 'gain' или 'split'

        Returns:
            DataFrame с колонками [feature, importance_return_1d, importance_return_20d, ...]
        """
        if self.model_return_1d is None:
            raise ValueError("Model not trained yet")

        importance_df = pd.DataFrame({
            'feature': self.feature_names
        })

        # Importance для regression моделей
        importance_df['importance_return_1d'] = self.model_return_1d.feature_importances_
        importance_df['importance_return_20d'] = self.model_return_20d.feature_importances_

        # Importance для калиброванных классификаторов (берем из базового классификатора)
        if self.model_prob_up_1d is not None:
            # CalibratedClassifierCV.calibrated_classifiers_ — список калибраторов
            base_clf_1d = self.model_prob_up_1d.calibrated_classifiers_[0].estimator
            importance_df['importance_prob_up_1d'] = base_clf_1d.feature_importances_

        if self.model_prob_up_20d is not None:
            base_clf_20d = self.model_prob_up_20d.calibrated_classifiers_[0].estimator
            importance_df['importance_prob_up_20d'] = base_clf_20d.feature_importances_

        # Средняя importance
        importance_cols = [col for col in importance_df.columns if col.startswith('importance_')]
        importance_df['importance_mean'] = importance_df[importance_cols].mean(axis=1)

        # Сортируем по средней importance
        importance_df = importance_df.sort_values('importance_mean', ascending=False)

        return importance_df
