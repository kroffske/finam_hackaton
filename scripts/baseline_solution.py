"""
Baseline решение для соревнования.

Простое решение без использования ML:
- Используем скользящее среднее последних N дней
- Предсказываем направление на основе моментума
- Это baseline для сравнения с более сложными решениями
"""

from pathlib import Path

import numpy as np
import pandas as pd


class BaselineSolution:
    """
    Baseline решение на основе скользящих средних и моментума

    Логика:
    1. Для каждого тикера вычисляем моментум (изменение цены за последние N дней)
    2. Предсказываем, что тренд продолжится (momentum continuation)
    3. Вероятность роста = сигмоида от моментума
    """

    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: Размер окна для вычисления моментума
        """
        self.window_size = window_size

    def load_data(self, train_candles_path: str,
                  public_test_path: str,
                  private_test_path: str):
        """Загрузка данных"""
        print("📊 Загрузка данных...")

        self.train_df = pd.read_csv(train_candles_path)
        self.train_df['begin'] = pd.to_datetime(self.train_df['begin'])

        public_test_df = pd.read_csv(public_test_path)
        public_test_df['begin'] = pd.to_datetime(public_test_df['begin'])

        private_test_df = pd.read_csv(private_test_path)
        private_test_df['begin'] = pd.to_datetime(private_test_df['begin'])

        # Объединяем оба теста
        self.test_df = pd.concat([public_test_df, private_test_df], ignore_index=True)

        # Объединяем для вычисления моментума (нужна история)
        self.full_df = pd.concat([self.train_df, self.test_df], ignore_index=True)
        self.full_df = self.full_df.sort_values(['ticker', 'begin'])

        print(f"   ✓ Train: {len(self.train_df)} строк")
        print(f"   ✓ Public test:  {len(public_test_df)} строк")
        print(f"   ✓ Private test: {len(private_test_df)} строк")
        print(f"   ✓ Total test:   {len(self.test_df)} строк")

    def compute_features(self):
        """Вычисление признаков (моментум, волатильность)"""
        print("\n🔧 Вычисление признаков...")

        df = self.full_df.copy()

        # Группируем по тикерам
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df[mask].copy()

            # 1. Моментум = процентное изменение цены за window_size дней
            ticker_data['momentum'] = (
                ticker_data['close'].pct_change(self.window_size)
            )

            # 2. Волатильность = std доходностей за window_size дней
            ticker_data['volatility'] = (
                ticker_data['close'].pct_change().rolling(self.window_size).std()
            )

            # 3. Средняя цена за window_size дней
            ticker_data['ma'] = ticker_data['close'].rolling(self.window_size).mean()

            # 4. Расстояние от MA (нормализованное)
            ticker_data['distance_from_ma'] = (
                (ticker_data['close'] - ticker_data['ma']) / ticker_data['ma']
            )

            # Обновляем данные
            df.loc[mask, 'momentum'] = ticker_data['momentum'].values
            df.loc[mask, 'volatility'] = ticker_data['volatility'].values
            df.loc[mask, 'ma'] = ticker_data['ma'].values
            df.loc[mask, 'distance_from_ma'] = ticker_data['distance_from_ma'].values

        self.full_df = df
        print("   ✓ Признаки вычислены")

    def predict(self):
        """
        Создание предсказаний

        Baseline стратегия:
        - pred_return = momentum * scaling_factor
        - pred_prob_up = sigmoid(momentum * sensitivity)
        """
        print("\n🎯 Создание предсказаний...")

        # Фильтруем только test данные
        test_data = self.full_df[
            self.full_df['begin'].isin(self.test_df['begin'])
        ].copy()

        # Заполняем NaN нулями (для первых строк где нет истории)
        test_data['momentum'] = test_data['momentum'].fillna(0)
        test_data['volatility'] = test_data['volatility'].fillna(0.01)
        test_data['distance_from_ma'] = test_data['distance_from_ma'].fillna(0)

        # Предсказание доходности
        # Простая стратегия: предсказываем что моментум продолжится
        # Для 1 дня: momentum * 0.3 (ослабляем сигнал)
        # Для 20 дней: momentum * 1.0 (накопленный эффект)
        test_data['pred_return_1d'] = test_data['momentum'] * 0.3
        test_data['pred_return_20d'] = test_data['momentum'] * 1.0

        # Предсказание вероятности роста
        # Используем сигмоиду для преобразования моментума в вероятность
        def sigmoid(x, sensitivity=10):
            return 1 / (1 + np.exp(-sensitivity * x))

        test_data['pred_prob_up_1d'] = sigmoid(test_data['momentum'], sensitivity=10)
        test_data['pred_prob_up_20d'] = sigmoid(test_data['momentum'], sensitivity=5)

        # Clipping: вероятности в диапазоне [0.1, 0.9] для стабильности
        test_data['pred_prob_up_1d'] = test_data['pred_prob_up_1d'].clip(0.1, 0.9)
        test_data['pred_prob_up_20d'] = test_data['pred_prob_up_20d'].clip(0.1, 0.9)

        # Clipping: доходности в разумном диапазоне [-0.2, 0.2]
        test_data['pred_return_1d'] = test_data['pred_return_1d'].clip(-0.2, 0.2)
        test_data['pred_return_20d'] = test_data['pred_return_20d'].clip(-0.5, 0.5)

        self.predictions = test_data

        print(f"   ✓ Создано {len(self.predictions)} предсказаний")
        print(f"\n   📊 Статистика предсказаний:")
        print(f"      Средняя pred_return_1d:  {test_data['pred_return_1d'].mean():.6f}")
        print(f"      Средняя pred_return_20d: {test_data['pred_return_20d'].mean():.6f}")
        print(f"      Средняя pred_prob_up_1d: {test_data['pred_prob_up_1d'].mean():.4f}")
        print(f"      Средняя pred_prob_up_20d: {test_data['pred_prob_up_20d'].mean():.4f}")

    def save_submission(self, output_path: str = "submission.csv"):
        """Сохранение submission файла"""
        print(f"\n💾 Сохранение submission...")

        submission = self.predictions[[
            'ticker', 'begin',
            'pred_return_1d', 'pred_return_20d',
            'pred_prob_up_1d', 'pred_prob_up_20d'
        ]].copy()

        submission.to_csv(output_path, index=False)

        print(f"   ✓ Submission сохранен: {output_path}")
        print(f"   Строк: {len(submission)}")
        print(f"\n   📋 Первые строки:")
        print(submission.head(10).to_string(index=False))

    def run(self, train_path: str, public_test_path: str,
            private_test_path: str, output_path: str = "submission.csv"):
        """Полный пайплайн baseline решения"""
        print("=" * 70)
        print("🚀 BASELINE РЕШЕНИЕ")
        print("=" * 70 + "\n")

        # 1. Загрузка данных
        self.load_data(train_path, public_test_path, private_test_path)

        # 2. Вычисление признаков
        self.compute_features()

        # 3. Предсказание
        self.predict()

        # 4. Сохранение
        self.save_submission(output_path)

        print("\n" + "=" * 70)
        print("✅ BASELINE ГОТОВ!")
        print("=" * 70)
        print(f"\n💡 Следующие шаги:")
        print(f"   1. Оцените на public:  python scripts/evaluate_submission.py {output_path} public")
        print(f"   2. Оцените на private: python scripts/evaluate_submission.py {output_path} private")
        print(f"   3. Используйте это решение как отправную точку для улучшений")
        print(f"   4. Попробуйте добавить ML модели, NLP для новостей и т.д.")


if __name__ == "__main__":
    baseline = BaselineSolution(window_size=5)

    baseline.run(
        train_path="data/raw/participants/train_candles.csv",
        public_test_path="data/raw/participants/public_test_candles.csv",
        private_test_path="data/raw/participants/private_test_candles.csv",
        output_path="baseline_submission.csv"
    )

