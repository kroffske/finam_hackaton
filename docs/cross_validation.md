# 🔄 Cross-Validation для временных рядов

> **Цель**: Надежная оценка моделей без data leakage при работе с перекрывающимися таргетами t+20

---

## ⚠️ Проблема: обычный K-Fold не работает!

### Почему обычная кроссвалидация опасна?

**Обычный K-Fold:**
```
Fold 1: [Train: samples 1-800  ] [Test: samples 801-1000]
Fold 2: [Train: samples 1-600  ] [Test: samples 601-800 ]
         ↑                              ↑
         Перемешаны по времени → DATA LEAKAGE!
```

**Проблемы:**
1. ❌ **Нарушение временного порядка** — модель видит будущее
2. ❌ **Overlap в таргетах** — `target_return_20d` в день t использует цену из дня t+20
3. ❌ **Нереалистичная оценка** — в продакшене мы НЕ знаем будущее

### Пример data leakage

```python
# День 980 (последний в train):
date_980 = '2025-01-06'
close_980 = 100.0
close_1000 = 105.0  # Через 20 торговых дней
target_return_20d = (105.0 / 100.0) - 1 = 0.05  # +5%

# День 1001 (первый в test):
date_1001 = '2025-02-04'

# УТЕЧКА: target_return_20d для дня 980 использует цену дня 1000,
# который находится ВНУТРИ test периода!
```

**Результат:** модель переобучается, метрики завышены, на продакшене работает хуже.

---

## 📊 Анализ данных: биржевые дни и пропуски

### Структура временных рядов

**Основные факты:**
- **Период**: 2020-06-19 до 2025-04-15
- **Всего**: 1,761 календарных дней
- **Торговых дней**: 1,217 (69% от календарных)
- **Тикеров**: 19

### Типы пропусков

#### 1. Выходные (регулярные)
```
2020-06-19 (Пт) → 2020-06-22 (Пн): 3 календарных дня
2020-06-26 (Пт) → 2020-06-29 (Пн): 3 календарных дня
```

**Частота**: каждую неделю
**Формула**: пт → пн = 3 дня (сб, вс)

#### 2. Праздники (нерегулярные)
```
2020-12-30 → 2021-01-04: 5 дней  (Новый год)
2022-04-29 → 2022-05-04: 5 дней  (Майские праздники)
2024-03-07 → 2024-03-11: 4 дня   (8 марта)
```

**Частота**: ~10 раз в год
**Размер**: 4-5 дней

#### 3. Аномалии (редкие)
```
2022-02-25 → 2022-03-24: 27 дней  (Приостановка торгов)
```

**Причина**: Геополитические события (закрытие биржи в феврале-марте 2022)

### Пропуски по тикерам

| Ticker | Полнота | Missing days | Примечания |
|--------|---------|--------------|------------|
| SBER   | 100%    | 0            | Полные данные |
| GAZP   | 100%    | 0            | Полные данные |
| T      | 98.19%  | 22           | Самые неполные |
| GMKN   | 99.67%  | 4            | Несколько пропусков |
| SIBN   | 99.51%  | 6            | Несколько пропусков |

**Вывод**: Большинство тикеров имеют полные данные, но есть 6 тикеров с пропусками.

---

## 📅 Что такое t+20 в торговых днях?

### Определение

**t+20** = 20 **торговых** дней вперед (НЕ календарных!)

```python
# Торговые дни (пропускаем выходные и праздники):
t = 2024-01-15 (Mon)
t+1 = 2024-01-16 (Tue)
t+2 = 2024-01-17 (Wed)
t+3 = 2024-01-18 (Thu)
t+4 = 2024-01-19 (Fri)
# Выходные: 2024-01-20 (Sat), 2024-01-21 (Sun) - ПРОПУСКАЕМ
t+5 = 2024-01-22 (Mon)
...
t+20 = 2024-02-12 (Mon)  # Примерно 28 календарных дней
```

### Формула конверсии

**20 торговых дней ≈ 28-30 календарных дней**

```
Календарные дни = Торговые дни × (7 / 5) × (1 + праздники)
                ≈ 20 × 1.4 × 1.02
                ≈ 28.6 дней
```

**Диапазон**: от 28 дней (обычные недели) до 35 дней (с длинными праздниками)

### Как определить t+20?

**Алгоритм:**
1. Получить отсортированный список всех торговых дат
2. Найти индекс текущей даты
3. Взять дату с индексом `current_index + 20`

```python
trading_dates = sorted(df['begin'].dt.date.unique())
# ['2020-06-19', '2020-06-22', '2020-06-23', ..., '2025-04-15']

current_date = datetime.date(2024, 1, 15)
current_idx = trading_dates.index(current_date)  # Например, 892

t_plus_20 = trading_dates[current_idx + 20]  # trading_dates[912]
# → 2024-02-12
```

### Примеры расчета

| t (начало) | t+20 (торговый) | Календарных дней | Примечание |
|------------|-----------------|------------------|------------|
| 2020-06-19 | 2020-07-21      | 32               | Включает выходные |
| 2020-09-01 | 2020-09-29      | 28               | Обычный месяц |
| 2020-11-11 | 2020-12-09      | 28               | Без праздников |
| 2024-01-15 | 2024-02-12      | 28               | Обычный период |
| 2024-03-01 | 2024-04-01      | 31               | Включает праздник 8 марта |

**Вывод**: В среднем 28-29 дней, но может варьироваться от 28 до 35 в зависимости от праздников.

---

## 🔧 Решение: Rolling Window CV с gap

### Принцип

**Ключевая идея:** Между train и test должен быть **зазор (gap) = 21 торговый день**

```
[────────── Train ──────────] [── Gap ──] [── Test ──]
    1,000 торговых дней         21 дней     60 дней
                              ↑
                        Защита от leakage!
```

**Почему 21?**
- 20 дней для `target_return_20d`
- +1 день safety margin
- Гарантирует: последний таргет в train не пересекается с test данными

### Схема разбиения (5 фолдов)

Общее количество торговых дней: **1,217**

```
Total:    [═══════════════════════════ 1,217 дней ═══════════════════════════]

Fold 1:   [════════════ Train ════════════][Gap][Test 60]
          Day 1 ────────────────→ Day 1115  1136  1157-1217

Fold 2:   [═══════ Train ════════][Gap][Test 60]
          Day 1 ──────────→ Day 1055  1076  1097-1157

Fold 3:   [══════ Train ══════][Gap][Test 60]
          Day 1 ────────→ Day 995   1016  1037-1097

Fold 4:   [═══ Train ════][Gap][Test 60]
          Day 1 ──→ Day 935   956   977-1037

Fold 5:   [═ Train ═][Gap][Test 60]
          Day 1 → Day 875   896   917-977
```

### Математика разбиения

**Параметры:**
- `n_splits = 5` — количество фолдов
- `test_size = 60` — размер test в торговых днях
- `gap = 21` — зазор между train и test

**Формула:**
```python
total_days = 1217
test_size = 60
gap = 21

for fold_idx in range(n_splits):
    # Индексы test
    test_end = total_days - fold_idx * test_size
    test_start = test_end - test_size + 1

    # Индексы train (с учетом gap)
    train_end = test_start - gap - 1
    train_start = 0

    print(f"Fold {fold_idx+1}:")
    print(f"  Train: day {train_start} to {train_end}")
    print(f"  Gap:   {gap} days")
    print(f"  Test:  day {test_start} to {test_end}")
```

**Результат:**
```
Fold 1: Train 1-1136 (1136 дней) | Gap 21 | Test 1157-1217 (60 дней)
Fold 2: Train 1-1076 (1076 дней) | Gap 21 | Test 1097-1157 (60 дней)
Fold 3: Train 1-1016 (1016 дней) | Gap 21 | Test 1037-1097 (60 дней)
Fold 4: Train 1-956  (956 дней)  | Gap 21 | Test 977-1037  (60 дней)
Fold 5: Train 1-896  (896 дней)  | Gap 21 | Test 917-977   (60 дней)
```

### Проверка data leakage

**Для каждого fold:**

```python
# Пример: Fold 1
train_end = day 1136 (например, 2025-01-06)
test_start = day 1157 (например, 2025-02-05)

# Проверка:
# 1. Последний день train (1136) имеет target_20d вычисленный из дня 1156
#    День 1156 = 2025-02-04 (за 1 день до test_start)
# 2. Первый день test (1157) = 2025-02-05
# 3. Нет пересечения ✓

gap_actual = test_start - train_end - 1 = 1157 - 1136 - 1 = 20
# Но мы используем gap=21 для safety → еще лучше!
```

**Вывод:** С gap=21 гарантированно НЕТ утечки данных.

---

## 💻 Реализация: src/finam/cv.py

### Основные функции

#### 1. `get_trading_dates(df)` — Получить список торговых дней
```python
def get_trading_dates(df: pd.DataFrame) -> list[datetime.date]:
    """
    Возвращает отсортированный список уникальных торговых дат

    Args:
        df: DataFrame с колонкой 'begin' (datetime)

    Returns:
        Список дат в формате datetime.date, отсортированный по возрастанию

    Example:
        >>> dates = get_trading_dates(train_df)
        >>> dates[:5]
        [datetime.date(2020, 6, 19),
         datetime.date(2020, 6, 22),
         datetime.date(2020, 6, 23), ...]
    """
```

#### 2. `compute_t_plus_n(df, date, n)` — Вычислить t+N
```python
def compute_t_plus_n(
    df: pd.DataFrame,
    date: datetime.date,
    n: int = 20
) -> datetime.date:
    """
    Вычисляет t+N в торговых днях

    Args:
        df: DataFrame с торговыми данными
        date: Начальная дата
        n: Количество торговых дней вперед (default: 20)

    Returns:
        Дата через N торговых дней

    Raises:
        ValueError: если date не найдена или t+n выходит за границы данных

    Example:
        >>> t_plus_20 = compute_t_plus_n(df, datetime.date(2024, 1, 15), n=20)
        >>> t_plus_20
        datetime.date(2024, 2, 12)  # Примерно 28 календарных дней
    """
```

#### 3. `rolling_window_cv(...)` — Основная функция CV
```python
def rolling_window_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = 60,
    gap: int = 21,
    min_train_size: int = 200
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Rolling window cross-validation с gap для временных рядов

    Args:
        df: DataFrame с торговыми данными (должен иметь колонку 'begin')
        n_splits: Количество фолдов (default: 5)
        test_size: Размер test в торговых днях (default: 60)
        gap: Зазор между train и test в торговых днях (default: 21)
        min_train_size: Минимальный размер train (default: 200)

    Yields:
        (train_df, test_df) для каждого фолда

    Example:
        >>> for fold_idx, (train, test) in enumerate(rolling_window_cv(df)):
        ...     print(f"Fold {fold_idx}: train={len(train)}, test={len(test)}")
        ...     model.fit(train)
        ...     metrics = model.evaluate(test)
    """
```

#### 4. `evaluate_with_cv(...)` — Оценка модели
```python
def evaluate_with_cv(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
    verbose: bool = True,
    **cv_kwargs
) -> dict[str, list[float]]:
    """
    Оценка модели с помощью кроссвалидации

    Args:
        model: Объект модели с методами fit() и predict()
        df: DataFrame с данными
        feature_cols: Список названий признаков
        n_splits: Количество фолдов
        verbose: Печатать прогресс
        **cv_kwargs: Дополнительные аргументы для rolling_window_cv()

    Returns:
        Dict с метриками для каждого фолда:
        {
            'mae_1d': [fold1, fold2, fold3, fold4, fold5],
            'mae_20d': [...],
            'brier_1d': [...],
            'brier_20d': [...],
            'da_1d': [...],
            'da_20d': [...],
            'score_1d': [...],
            'score_20d': [...],
            'score_total': [...]
        }

    Example:
        >>> from finam.model import LightGBMModel
        >>> model = LightGBMModel()
        >>> cv_results = evaluate_with_cv(
        ...     model, train_df, feature_cols,
        ...     n_splits=5, test_size=60, gap=21
        ... )
        >>> print(f"Mean MAE 1d: {np.mean(cv_results['mae_1d']):.4f}")
        >>> print(f"Std MAE 1d:  {np.std(cv_results['mae_1d']):.4f}")
    """
```

### Пример использования

```python
from finam.cv import rolling_window_cv, evaluate_with_cv
from finam.model import LightGBMModel
import pandas as pd

# 1. Загрузка данных
train_df = pd.read_parquet('data/preprocessed/train.parquet')

# 2. Получить список признаков
feature_cols = [col for col in train_df.columns
                if col not in ['ticker', 'begin', 'open', 'high', 'low',
                               'close', 'volume', 'adj_close']
                and not col.startswith('target_')]

# 3. Создать модель
model = LightGBMModel(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7
)

# 4. Оценить с помощью CV
cv_results = evaluate_with_cv(
    model=model,
    df=train_df,
    feature_cols=feature_cols,
    n_splits=5,
    test_size=60,
    gap=21,
    verbose=True
)

# 5. Анализ результатов
import numpy as np

print("=" * 60)
print("CROSS-VALIDATION RESULTS")
print("=" * 60)

for metric_name in ['mae_1d', 'mae_20d', 'score_total']:
    values = cv_results[metric_name]
    print(f"{metric_name:15s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    print(f"                   Range: [{np.min(values):.4f}, {np.max(values):.4f}]")

# 6. Сохранить результаты
import json
with open('cv_results.json', 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in cv_results.items()}, f, indent=2)
```

**Вывод:**
```
============================================================
CROSS-VALIDATION RESULTS
============================================================
mae_1d         : 0.0172 ± 0.0013
                   Range: [0.0156, 0.0188]
mae_20d        : 0.0895 ± 0.0067
                   Range: [0.0821, 0.0971]
score_total    : 0.1055 ± 0.0089
                   Range: [0.0942, 0.1167]
```

---

## 📐 Выбор параметров CV

### Test size (размер test)

**Рекомендация: 60 торговых дней**

**Почему?**
- 60 дней ≈ 3 месяца торговли
- Достаточно большой для статистически значимой оценки
- Не слишком большой (чтобы было место для нескольких фолдов)
- Соответствует реальным циклам отчетности (квартал)

**Варианты:**
- **40 дней** (2 месяца) — для более частых фолдов
- **60 дней** (3 месяца) — рекомендуется ✓
- **80 дней** (4 месяца) — для более долгосрочной оценки

### Gap (зазор)

**Рекомендация: 21 торговый день**

**Почему?**
- 20 дней для `target_return_20d`
- +1 день safety margin
- Гарантирует отсутствие data leakage

**Варианты:**
- **20 дней** — минимум (ровно для t+20)
- **21 день** — рекомендуется ✓
- **25 дней** — extra safety (если есть сомнения)

### Number of splits (количество фолдов)

**Рекомендация: 5 фолдов**

**Почему?**
- Баланс между computational cost и надежностью
- Каждый fold тренируется на 800-1100 днях (достаточно)
- Standard в ML (5-fold или 10-fold)

**Варианты:**
- **3 фолда** — быстрее, но менее надежная оценка
- **5 фолдов** — рекомендуется ✓
- **10 фолдов** — более надежно, но дольше считается

### Формула расчета

```python
total_days = 1217  # Доступно торговых дней

# Требуемое количество дней для n_splits фолдов:
required_days = n_splits * test_size + (n_splits - 1) * gap + min_train_size

# Пример для n_splits=5:
required = 5 * 60 + 4 * 21 + 200 = 300 + 84 + 200 = 584 дней

# Доступно: 1217 > 584 ✓
# Можно использовать 5 фолдов!
```

---

## ⚠️ Особые случаи и ограничения

### 1. Missing ticker data

**Проблема:** Некоторые тикеры (T, GMKN, SIBN) имеют пропущенные даты

**Решения:**

**Вариант A: Drop incomplete tickers**
```python
# Убрать тикеры с >1% пропусков
completeness = df.groupby('ticker')['begin'].nunique() / len(trading_dates)
complete_tickers = completeness[completeness >= 0.99].index
df_filtered = df[df['ticker'].isin(complete_tickers)]
```

**Вариант B: Forward-fill missing dates**
```python
# Заполнить пропуски последней известной ценой
df = df.set_index(['ticker', 'begin']).sort_index()
df = df.groupby('ticker').fillna(method='ffill')
```

**Вариант C: Skip missing dates (рекомендуется для CV)**
```python
# Просто игнорировать отсутствующие даты
# CV автоматически использует только доступные данные
```

### 2. Период закрытия биржи (2022-02-25 → 2022-03-24)

**Проблема:** 27-дневный gap нарушает нормальную структуру данных

**Решения:**

**Вариант A: Exclude period**
```python
# Убрать этот период из данных
df = df[(df['begin'] < '2022-02-25') | (df['begin'] > '2022-03-24')]
```

**Вариант B: Treat as a regime change**
```python
# Добавить dummy variable для этого периода
df['is_crisis_period'] = (df['begin'] >= '2022-02-25') & (df['begin'] <= '2022-03-24')
```

**Вариант C: Keep as is (рекомендуется)**
- Модель должна быть робастной к таким событиям
- В реальной торговле такое может повториться
- Просто помнить что метрики могут быть хуже на этом периоде

### 3. Последние дни без таргетов

**Проблема:** Последние 20 дней каждого тикера не имеют `target_return_20d`

**Решение:** Уже обработано в `scripts/1_prepare_data.py`:
```python
# Удаляем строки с NaN в таргетах
df.dropna(subset=['target_return_1d', 'target_return_20d'], inplace=True)
```

**Результат:** В preprocessed данных уже нет строк без таргетов ✓

---

## 📊 Интерпретация результатов CV

### Что означают метрики?

**Mean (среднее)** — среднее качество модели на разных периодах
```python
mean_mae = np.mean(cv_results['mae_1d'])
# Показывает типичную ошибку модели
```

**Std (стандартное отклонение)** — стабильность модели
```python
std_mae = np.std(cv_results['mae_1d'])
# Маленькое std = модель стабильна на разных периодах ✓
# Большое std = модель нестабильна, переобучена ✗
```

**Range (диапазон)** — разброс качества
```python
min_mae = np.min(cv_results['mae_1d'])
max_mae = np.max(cv_results['mae_1d'])
# Показывает худший/лучший фолд
```

### Пример интерпретации

```
MAE 1d: 0.0172 ± 0.0013  Range: [0.0156, 0.0188]
        ↑       ↑                 ↑       ↑
      Среднее  Std              Min     Max
```

**Вывод:**
- ✅ Средняя ошибка: 1.72% (хорошо!)
- ✅ Стабильность: ±0.13% (очень стабильна!)
- ✅ Худший фолд: 1.88% (приемлемо)
- ✅ Лучший фолд: 1.56% (отлично)

### Red flags (тревожные признаки)

**🚩 Большой std:**
```
MAE 1d: 0.0172 ± 0.0045  Range: [0.0120, 0.0250]
                  ↑↑↑↑
            Слишком большой! (26% от mean)
```
**Причины:**
- Модель переобучена
- Слишком сложная модель
- Нестабильные признаки

**Решение:** Упростить модель, добавить регуляризацию

**🚩 Противоречие train vs CV:**
```
Train MAE: 0.010  (отлично!)
CV MAE:    0.025  (плохо!)
           ↑↑↑↑
      Переобучение!
```

**Решение:** Уменьшить complexity модели

---

## 🎯 Best Practices

### 1. Всегда используйте CV перед финальной оценкой
```python
# ✓ ПРАВИЛЬНО:
cv_results = evaluate_with_cv(model, train_df, feature_cols)
if np.mean(cv_results['score_total']) > threshold:
    # Обучить на всех train данных
    model.fit(train_df, feature_cols)
    # Оценить на val/test
    test_metrics = model.evaluate(test_df)

# ✗ НЕПРАВИЛЬНО:
model.fit(train_df, feature_cols)  # Сразу обучили на всех данных
test_metrics = model.evaluate(test_df)  # Может быть завышено!
```

### 2. Проверяйте стабильность
```python
# Если std > 30% от mean → модель нестабильна
cv_results = evaluate_with_cv(model, train_df, feature_cols)
stability_ratio = np.std(cv_results['mae_1d']) / np.mean(cv_results['mae_1d'])

if stability_ratio > 0.3:
    print("⚠️ WARNING: Model is unstable!")
    print("   Consider: reduce model complexity, add regularization")
```

### 3. Сохраняйте результаты CV
```python
# Для каждого эксперимента сохранить CV метрики
import json

cv_summary = {
    'experiment_name': exp_name,
    'cv_params': {'n_splits': 5, 'test_size': 60, 'gap': 21},
    'metrics': {
        'mae_1d_mean': float(np.mean(cv_results['mae_1d'])),
        'mae_1d_std': float(np.std(cv_results['mae_1d'])),
        # ... другие метрики
    },
    'fold_results': cv_results  # Полные результаты всех фолдов
}

with open(f'outputs/{exp_name}/cv_results.json', 'w') as f:
    json.dump(cv_summary, f, indent=2)
```

### 4. Используйте CV для выбора гиперпараметров
```python
# Grid search с CV
best_score = -np.inf
best_params = None

for n_estimators in [100, 300, 500]:
    for learning_rate in [0.01, 0.05, 0.1]:
        model = LightGBMModel(n_estimators=n_estimators, learning_rate=learning_rate)
        cv_results = evaluate_with_cv(model, train_df, feature_cols)
        score = np.mean(cv_results['score_total'])

        if score > best_score:
            best_score = score
            best_params = {'n_estimators': n_estimators, 'learning_rate': learning_rate}

print(f"Best params: {best_params}")
```

---

## 🔗 Связанные документы

- **[docs/METRICS.md](METRICS.md)** — Описание метрик MAE, Brier, DA
- **[docs/evaluation.md](evaluation.md)** — Методика оценки соревнования
- **[CLAUDE.md](../CLAUDE.md)** — Guidelines для разработки
- **[README.md](../README.md)** — Обзор проекта и workflow

---

## 💡 Takeaways

1. ✅ **Gap = 21 день** обязателен для t+20 таргетов
2. ✅ **Test size = 60 дней** оптимален для баланса
3. ✅ **5 фолдов** достаточно для надежной оценки
4. ✅ **Торговые дни ≠ календарные дни** (20 торговых ≈ 28 календарных)
5. ✅ **Std < 30% от mean** → модель стабильна
6. ✅ **Всегда CV перед финальной оценкой** для избежания переобучения

**Главное правило:** Временные ряды требуют gap между train и test!
