# AI Agent Guidelines - finam FORECAST

**Goal:** Быстрое решение для хакатона. Максимальная простота, минимум dependencies, максимальная скорость итераций.

---

## 📋 Quick Checklist

- ✅ **Простота**: Один файл = одна задача. Без overengineering.
- ✅ **Baseline First**: Сначала работает простое, потом улучшаем.
- ✅ **Quality**: `ruff check --fix && ruff format` перед коммитом.
- ✅ **Colab Ready**: Можно установить через `pip install git+https://...`
- ✅ **Explicit Args**: Функции с явными параметрами, без Hydra/configs.

---

## 🎯 Задача FORECAST (кратко)

**Input:** Цены и новости до дня `t` включительно
**Output:** Для каждого актива:
- `pred_return_1d` — прогноз доходности на 1 день
- `pred_return_20d` — прогноз доходности на 20 дней
- `pred_prob_up_1d` — вероятность роста за 1 день (генерируется через sigmoid)
- `pred_prob_up_20d` — вероятность роста за 20 дней (генерируется через sigmoid)

**Метрика:**
```
Score = MAE (Mean Absolute Error)
```
где:
- `MAE` = Mean Absolute Error для доходности (меньше = лучше)
- Усредняется по обоим горизонтам (1d и 20d)

**Ограничения:**
- Цены доступны до `t` включительно
- Новости доступны до `t-1` (задержка 1 день)
- Время работы train+predict ≤ 60 минут
- Воспроизводимость: фиксированный seed

**Референс:** `scripts/baseline_solution.py` — простой momentum-based подход

---

## 🏗️ Целевая архитектура src/finam/

Простая структура без излишней вложенности:

```
src/finam/
├── __init__.py           # package marker
├── metrics.py            # MAE, Brier, DA, normalized scores
├── features.py           # technical indicators (momentum, volatility, MA, RSI, MACD, etc)
├── features_news.py      # ✨ NEWS features (counts с правильным lag)
├── model.py              # model wrapper (MomentumBaseline, LightGBMModel)
├── evaluate.py           # model comparison utilities
└── cv.py                 # cross-validation (rolling window для временных рядов)
```

### Описание модулей

**metrics.py** — расчёт метрик (только MAE)
```python
def mae(y_true, y_pred) -> float
def evaluate_predictions(y_true_1d, y_true_20d, pred_1d, pred_20d) -> dict
def print_metrics(metrics: dict, model_name: str) -> None
```

**features.py** — генерация признаков
```python
def compute_momentum(df: pd.DataFrame, window: int = 5) -> pd.DataFrame
def compute_volatility(df: pd.DataFrame, window: int = 5) -> pd.DataFrame
def compute_moving_average(df: pd.DataFrame, window: int = 5) -> pd.DataFrame
def add_all_features(df: pd.DataFrame, windows: list[int] = [5, 20]) -> pd.DataFrame
```
> Референс: `scripts/baseline_solution.py:60-96`

**model.py** — обёртка над моделями (только regression)
```python
class BaseModel:
    def fit(X, y_return_1d, y_return_20d) -> None
    def predict(X) -> dict  # returns {pred_return_1d, pred_return_20d}

class MomentumBaseline(BaseModel):  # baseline из scripts/
class LightGBMModel(BaseModel):     # 2 regression модели (MAE loss)
```

**pipeline.py** — train/predict workflow
```python
def train_pipeline(train_candles_path, output_model_path, **kwargs) -> None
def predict_pipeline(test_candles_path, model_path, output_submission_path) -> None
```

**cv.py** — кросс-валидация для временных рядов
```python
def rolling_cv_split(df: pd.DataFrame, n_splits: int = 5, test_size: int = 20) -> Iterator
def evaluate_model_cv(model, df: pd.DataFrame, cv_splitter) -> dict[str, float]
```

---

## 🚀 Workflow для LLM агента

### 1. Analyze (Анализ задачи)
- Прочитать baseline в `scripts/baseline_solution.py`
- Понять формат данных, метрики, pipeline
- Определить scope изменений

### 2. Plan (Составить план)
```
1. Какую гипотезу проверяем?
2. Какой модуль создаём/меняем?
3. Какие признаки добавляем?
4. Как измерим улучшение?
```

### 3. Code (Реализация)
- Создать/изменить один файл за раз
- Явные параметры функций (не глобальные конфиги)
- Простые интерфейсы (pandas DataFrames in/out)
- Добавить docstrings с примерами

### 4. Test (Проверка)
```bash
# Lint и форматирование
ruff check --fix src/ && ruff format src/

# Запустить baseline для проверки
python scripts/baseline_solution.py

# Запустить свой код (когда будет готов)
python -m src.finam.pipeline train ...
python -m src.finam.pipeline predict ...
```

### 5. Document (Запись в SESSION.md)
```
## 2025-10-03: Добавил модуль features.py
- Реализованы функции для momentum, volatility, MA
- Базируется на scripts/baseline_solution.py:60-96
- Готово для интеграции в pipeline
```

---

## ⚡ Golden Rules для Хакатона

### 1. Простота > Сложность
```python
# ✅ GOOD: Явные параметры
def compute_momentum(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    return df.groupby('ticker')['close'].pct_change(window)

# ❌ BAD: Скрытые зависимости
def compute_momentum(df: pd.DataFrame):
    window = CONFIG['features']['momentum_window']  # откуда CONFIG?
    ...
```

### 2. Baseline First
- Сначала воспроизвести baseline из scripts/
- Потом добавлять улучшения по одному
- Измерять каждое изменение

### 3. Explicit > Implicit
```python
# ✅ GOOD: Понятно что на входе/выходе
def train_model(train_df: pd.DataFrame, feature_cols: list[str],
                target_col: str = 'return_1d') -> BaseModel:
    ...

# ❌ BAD: Неясно что происходит
def train_model(cfg):  # что в cfg? какие поля? какие типы?
    ...
```

### 4. Быстрая обратная связь
```bash
# Частые коммиты с маленькими изменениями
git add src/finam/metrics.py
git commit -m "feat: add MAE and Brier metrics"

# Быстрая проверка
ruff check src/ && python -m pytest tests/ -q
```

### 5. Colab Compatibility
```python
# ✅ GOOD: Работает в Colab
!pip install git+https://github.com/user/finam.git
from finam.model import LightGBMModel
from finam.features import add_all_features

# ✅ GOOD: Minimal dependencies
# pandas, numpy, scikit-learn — уже в Colab
# lightgbm, joblib, pyyaml — легковесные
```

---

## 🔧 Примеры использования

### Быстрый старт в Colab
```python
# 1. Установка
!pip install git+https://github.com/your-repo/finam.git

# 2. Импорты
from finam.features import add_all_features
from finam.model import MomentumBaseline
from finam.metrics import mae, brier_score, normalized_score

# 3. Загрузка данных
import pandas as pd
train_df = pd.read_csv('train_candles.csv')
test_df = pd.read_csv('test_candles.csv')

# 4. Feature engineering
train_df = add_all_features(train_df, windows=[5, 20])
test_df = add_all_features(test_df, windows=[5, 20])

# 5. Подготовка таргетов
X_train = train_df[feature_cols]
y_return_1d = train_df['target_return_1d'].values
y_return_20d = train_df['target_return_20d'].values

# 6. Train (только regression для MAE)
model = LightGBMModel()
model.fit(X_train, y_return_1d, y_return_20d)

# 7. Predict
X_test = test_df[feature_cols]
predictions = model.predict(X_test)  # {pred_return_1d, pred_return_20d}

# 8. Evaluate
from finam.metrics import evaluate_predictions
metrics = evaluate_predictions(
    y_true_1d, y_true_20d,
    predictions['pred_return_1d'],
    predictions['pred_return_20d']
)
print(f"MAE 1d: {metrics['mae_1d']:.6f}")
print(f"MAE 20d: {metrics['mae_20d']:.6f}")
print(f"MAE mean: {metrics['mae_mean']:.6f}")
```

---

## 🚫 Анти-паттерны для Хакатона

### ❌ Не делать:
- **Hydra configs** — слишком сложно для простого пайплайна
- **Многоуровневые абстракции** — core/formatting/orchestration излишни
- **Pydantic models** — достаточно простых функций с типами
- **Сложная структура проектов** — держим flat hierarchy
- **OOP без необходимости** — функции проще для Jupyter/Colab

### ✅ Делать:
- **Функции с явными параметрами** — легко тестировать в ноутбуке
- **Один файл = одна задача** — metrics, features, model отдельно
- **Pandas in/out** — стандарт для data science
- **Простые примеры в docstrings** — понятно как использовать
- **Измеримые улучшения** — каждое изменение = эксперимент с метриками

---

## 📚 Референсы

**Baseline решение:**
`scripts/baseline_solution.py` — momentum-based подход (простой, работает, можно улучшать)

**Документация задачи:**
- `docs/task.md` — полное описание задачи FORECAST
- `docs/evaluation.md` — метрики и формулы оценки
- `docs/data.md` — формат данных (если есть)

**Quality gates:**
```bash
ruff check --fix src/ tests/
ruff format src/ tests/
```

**Логирование:**
- `SESSION.md` — краткие записи о прогрессе (append only)
- `TODO.md` — текущие задачи (gitignored, для живой работы)

---

## 🎓 Для агента: стратегия улучшения baseline

### Итерация 1: Воспроизводимость
1. Перенести baseline из scripts/ в src/finam/
2. Проверить что получаются те же результаты
3. Добавить тесты на метрики

### Итерация 2: Feature Engineering
1. Добавить больше технических индикаторов (RSI, MACD, Bollinger Bands)
2. Добавить lag features для цен
3. Cross-ticker features (sector momentum, market regime)

### Итерация 3: Simple ML
1. Linear Regression / Ridge для калибровки
2. LightGBM для нелинейных паттернов
3. Ensemble (среднее по нескольким моделям)

### Итерация 4: Cross-Validation
1. Rolling window CV для временных рядов
2. Оптимизация гиперпараметров
3. Проверка стабильности на разных периодах

### Итерация 5: News Integration ✅ РЕАЛИЗОВАНО

**features_news.py** — новостные фичи с правильным lag

```python
def compute_daily_news_count(news_df, date_col='publish_date') -> pd.DataFrame
def add_news_features(candles_df, news_df, lag_days=1, rolling_windows=[1, 7, 30]) -> pd.DataFrame
```

**Реализованные фичи:**
- `news_count_1d_lag` — количество новостей за предыдущий день
- `news_count_7d_lag` — за последние 7 дней (с лагом)
- `news_count_30d_lag` — за последние 30 дней (с лагом)

**⚠️ ВАЖНО: Data Leakage Protection**
- Новости доступны до `t-1` (задержка 1 день)
- Для свечей дня `t` используем новости до `t-lag_days`
- Автоматический сдвиг дат для безопасного джойна

**Результаты:**
- ✅ Feature importance: топ-2 признака
- ✅ Улучшение MAE vs Momentum baseline

**Следующие шаги:**
1. Sentiment analysis (VADER, FinBERT)
2. Topic modeling (LDA, BERTopic)
3. Entity extraction (какие компании упоминаются)
4. Weighted average sentiment по тикерам

---

### Итерация 6: Упрощение под MAE ✅ РЕАЛИЗОВАНО

**Цель:** Упростить проект, убрав Brier Score, DA и калибровку.

**Изменения:**
- Удалены classification модели (prob_up_*)
- Удален класс `CalibratedLightGBMModel`
- Упрощены метрики: только MAE для 1d и 20d
- `pred_prob_up_*` генерируются через sigmoid в submission для совместимости с форматом
- Ускорение обучения: только 2 регрессора вместо 4 моделей

**Результат:**
- ✅ Упрощение кода на ~40%
- ✅ Фокус на главной метрике (MAE)
- ✅ Сохранение совместимости с submission форматом

---

**Главное правило:** Каждая итерация = +1 гипотеза, +код, +метрики. Без метрик = не считается.

Удачи! 🚀
