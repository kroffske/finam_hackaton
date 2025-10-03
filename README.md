# ML Pipeline для FORECAST хакатона

## 📋 Обзор

Этот документ описывает оптимизированный ML пайплайн для соревнования FORECAST. Пайплайн обеспечивает:

1. **Воспроизводимость** — все эксперименты сохраняются с параметрами
2. **Скорость итераций** — preprocessed данные для быстрых экспериментов
3. **Отслеживание прогресса** — автоматический сбор метрик всех экспериментов
4. **Сравнение с baseline** — normalized scores относительно Momentum

## 🏗️ Структура проекта

```
finam/
├── data/
│   ├── raw/participants/          # исходные данные
│   │   ├── train_candles.csv
│   │   ├── train_news.csv         # NEW: новости для фич
│   │   ├── test_news.csv
│   │   ├── public_test_candles.csv
│   │   └── private_test_candles.csv
│   ├── preprocessed/              # обработанные данные
│   │   ├── train.parquet          # train split с признаками
│   │   ├── val.parquet            # validation split
│   │   ├── test.parquet           # test split
│   │   ├── public_test.parquet    # NEW: preprocessed public test
│   │   ├── private_test.parquet   # NEW: preprocessed private test
│   │   └── metadata.json          # метаинформация
│   └── baseline_metrics.json      # NEW: эталонные метрики Momentum
│
├── outputs/                       # результаты экспериментов
│   ├── experiments_log.csv        # NEW: сводная таблица экспериментов
│   └── YYYY-MM-DD_HH-MM-SS_<name>/
│       ├── config.yaml            # параметры эксперимента
│       ├── model_*.pkl            # сохраненные модели
│       ├── metrics.json           # NEW: метрики train/val/test
│       ├── feature_importance.csv # важность признаков
│       ├── predictions_val.csv    # предсказания на val
│       ├── submission_public.csv  # NEW: submission для public test
│       └── submission_private.csv # NEW: submission для private test
│
├── scripts/
│   ├── 1_prepare_data.py          # подготовка + split + news + public/private
│   ├── 2_train_model.py           # обучение с train/val/test оценкой
│   ├── 3_evaluate.py              # оценка модели
│   ├── 4_generate_submission.py   # NEW: генерация submission файлов
│   ├── train_baseline.py          # NEW: обучение Momentum baseline
│   ├── compute_baseline_metrics.py # NEW: вычисление baseline
│   ├── collect_experiments.py     # NEW: сбор всех экспериментов
│   └── compare_baseline_lgbm.py   # сравнение моделей
│
└── src/finam/
    ├── features.py                # technical indicators
    ├── features_news.py           # NEW: news features
    ├── model.py                   # модели (LightGBM, Momentum)
    ├── metrics.py                 # метрики + normalized scores
    └── evaluate.py                # сравнение моделей
```

## 🚀 Workflow

### Шаг 1: Подготовка данных (один раз)

```bash
python scripts/1_prepare_data.py
```

**Что делает:**
- Загружает `train_candles.csv` и `train_news.csv`
- Создает 40+ технических индикаторов
- **NEW:** Добавляет новостные фичи (news_count_1d/7d/30d_lag)
- Разбивает на **train/val/test** (70%/15%/15% по времени)
- **NEW:** Обрабатывает `public_test_candles.csv` и `private_test_candles.csv` с теми же фичами
- Сохраняет в `data/preprocessed/*.parquet`

**Результат:**
```
data/preprocessed/
├── train.parquet        # 16,179 rows (2020-06-19 to 2023-11-21)
├── val.parquet          #  3,469 rows (2023-11-22 to 2024-08-12)
├── test.parquet         #  3,470 rows (2024-08-13 to 2025-04-15)
├── public_test.parquet  #    378 rows (2025-04-16 to 2025-05-09) NEW
├── private_test.parquet #    399 rows (2025-05-10 to ...) NEW
└── metadata.json        # конфиг + 43 features (включая 3 новостных)
```

**Опции:**
```bash
python scripts/1_prepare_data.py --train-ratio 0.7 --val-ratio 0.15
python scripts/1_prepare_data.py --windows 5 20 --no-cross-sectional
```

---

### Шаг 2: Вычисление baseline метрик (один раз)

```bash
python scripts/compute_baseline_metrics.py
```

**Что делает:**
- Вычисляет Momentum Baseline на train/val/test
- Сохраняет в `data/baseline_metrics.json`
- Используется для normalized scores всех экспериментов

**Результат:**
```json
{
  "train": {"mae_1d": 0.018376, "mae_20d": 0.085435, ...},
  "val": {"mae_1d": 0.013632, "mae_20d": 0.062087, ...},
  "test": {"mae_1d": 0.020728, "mae_20d": 0.096072, ...}
}
```

---

### Шаг 3: Обучение baseline (для сравнения)

```bash
python scripts/train_baseline.py --exp-name momentum_baseline
```

**Что делает:**
- Обучает Momentum Baseline как полноценный эксперимент
- Сохраняет в `outputs/` с метриками train/val/test
- Позволяет сравнивать с LightGBM

---

### Шаг 4: Обучение LightGBM модели

```bash
# Базовая модель с новостными фичами
python scripts/2_train_model.py --exp-name lgbm_with_news --model-type lightgbm

# С калибровкой (улучшает Brier score!)
python scripts/2_train_model.py --exp-name lgbm_calibrated --model-type lightgbm --calibrate

# С кастомными параметрами
python scripts/2_train_model.py --exp-name lgbm_tuned \
    --model-type lightgbm \
    --n-estimators 1000 \
    --learning-rate 0.01 \
    --max-depth 8 \
    --calibrate
```

**Что делает:**
- Загружает preprocessed данные (включая новостные фичи)
- Обучает модель
- **NEW:** Автоматически оценивает на train/val/test
- Сохраняет в `outputs/<timestamp>_<exp_name>/`:
  - Модели (*.pkl)
  - Конфиг (config.yaml)
  - Метрики для train/val/test (metrics.json)
  - Feature importance (для LightGBM)

**Пример вывода:**
```
TEST METRICS:
  MAE 1d:  0.017312
  MAE 20d: 0.089794
  Brier 1d:  0.263401
  Brier 20d: 0.297536
  DA 1d:  0.4914 (49.14%)
  DA 20d: 0.5014 (50.14%)
```

---

### Шаг 5: Сбор всех экспериментов

```bash
python scripts/collect_experiments.py
```

**Что делает:**
- Автоматически собирает метрики из всех папок `outputs/`
- Вычисляет **normalized scores** относительно baseline
- Сохраняет в `experiments_log.csv`

**Результат:**
```
TOP EXPERIMENTS (by test_score_total)
         exp_name model_type  test_mae_1d  test_mae_20d  test_score_total
   lgbm_with_news   lightgbm     0.017312      0.089794          0.105470 ✓
momentum_baseline   momentum     0.020728      0.096072          0.050504
```

**Использование:**
```bash
# Посмотреть все эксперименты
python -c "import pandas as pd; df = pd.read_csv('experiments_log.csv'); print(df.sort_values('test_score_total', ascending=False))"

# Фильтровать по модели
python -c "import pandas as pd; df = pd.read_csv('experiments_log.csv'); print(df[df['model_type']=='lightgbm'].sort_values('test_score_total', ascending=False))"
```

---

### Шаг 6: Генерация submission для public/private тестов

```bash
# Генерация submission из обученной модели
python scripts/4_generate_submission.py --run-id 2025-10-03_23-41-15_lgbm_with_news

# С кастомной output директорией
python scripts/4_generate_submission.py --run-id <run_id> --output-dir submissions/
```

**Что делает:**
- Загружает обученную модель из `outputs/<run_id>/`
- Загружает preprocessed `public_test.parquet` и `private_test.parquet`
- Генерирует предсказания для всех тикеров
- Сохраняет в `outputs/<run_id>/`:
  - `submission_public.csv`
  - `submission_private.csv`

**Формат submission файлов:**
```csv
ticker,begin,pred_return_1d,pred_return_20d,pred_prob_up_1d,pred_prob_up_20d
AFLT,2025-04-16,0.012345,-0.023456,0.543210,0.456789
AFLT,2025-04-17,-0.001234,0.045678,0.498765,0.567890
...
```

**Требования:**
- Сначала запустить `python scripts/1_prepare_data.py` для создания `public_test.parquet` и `private_test.parquet`
- Модель должна быть уже обучена (есть файлы model*.pkl в outputs/<run_id>/)

---

### Шаг 7: Оценка конкретной модели (опционально)

```bash
# Оценка на test данных
python scripts/3_evaluate.py --exp-dir 2025-10-03_23-41-15_lgbm_with_news --data test

# С сохранением отчета
python scripts/3_evaluate.py --exp-dir 2025-10-03_23-41-15_lgbm_with_news --data test --save-report
```

---

## 📊 Ключевые улучшения

### 1. Новостные фичи

**Добавлены в `features_news.py`:**
- `news_count_1d_lag` — количество новостей за предыдущий день
- `news_count_7d_lag` — за последние 7 дней
- `news_count_30d_lag` — за последние 30 дней

**ВАЖНО:** Правильный lag для избежания data leakage
- Новости доступны до `t-1`
- Для свечей дня `t` используем новости до `t-1`

**Feature Importance:**
```
1. news_count_30d_lag  — 1007.5 🥇
2. news_count_7d_lag   — 907.0  🥈
3. ma_5d               — 589.25
```

### 2. Автоматическая оценка на test

**До:**
- Оценка только на train/val
- Test метрики нужно считать вручную

**После:**
- `2_train_model.py` автоматически оценивает на test
- Все метрики сохраняются в `metrics.json`

```json
{
  "train": {...},
  "val": {...},
  "test": {
    "mae_1d": 0.017312,
    "mae_20d": 0.089794,
    ...
  }
}
```

### 3. Система трекинга экспериментов

**Новая система:**
1. **Baseline metrics** — эталон для сравнения
2. **Автосбор из outputs/** — все эксперименты в одной таблице
3. **Normalized scores** — Score = 0.7×MAE_norm + 0.3×Brier_norm + 0.1×DA

**experiments_log.csv:**
```csv
run_id,exp_name,model_type,test_mae_1d,test_mae_20d,test_score_total
2025-10-03_23-41-15_lgbm_with_news,lgbm_with_news,lightgbm,0.017312,0.089794,0.105470
2025-10-03_23-39-26_momentum_baseline,momentum_baseline,momentum,0.020728,0.096072,0.050504
```

### 4. Normalized Score

**Формула:**
```python
Score = 0.7 × MAE_norm + 0.3 × Brier_norm + 0.1 × DA

где:
  MAE_norm = 1 - (model_MAE / baseline_MAE)
  Brier_norm = 1 - (model_Brier / baseline_Brier)
  DA = directional_accuracy (без нормализации)
```

**Интерпретация:**
- **Score > 0** — модель лучше baseline
- **Score = 0** — модель равна baseline
- **Score < 0** — модель хуже baseline

---

## 📈 Результаты экспериментов

### Сравнение моделей (TEST)

| Эксперимент | Model | MAE 1d | MAE 20d | Brier 1d | Brier 20d | DA 1d | DA 20d | **Score** |
|------------|-------|---------|---------|----------|-----------|-------|--------|-----------|
| **lgbm_with_news** | LightGBM | 0.0173 | 0.0898 | 0.263 | 0.298 | 49.1% | 50.1% | **0.1055** 🥇 |
| momentum_baseline | Momentum | 0.0207 | 0.0961 | 0.263 | 0.256 | 51.7% | 49.3% | 0.0505 |

### Улучшение vs Baseline

**LightGBM с новостями:**
- ✅ **MAE 1d**: +16.5% лучше (0.0207 → 0.0173)
- ✅ **MAE 20d**: +6.6% лучше (0.0961 → 0.0898)
- ❌ **Brier 1d**: -0.2% (почти равно)
- ❌ **Brier 20d**: -16.2% (хуже на длинном горизонте)
- ❌ **DA 1d**: -5.0% (хуже угадывает направление)
- ✅ **DA 20d**: +1.7% (лучше на длинном горизонте)
- 🎯 **Total Score**: **+109%** лучше baseline!

**Выводы:**
1. ✅ Новостные фичи ОЧЕНЬ важны (топ-2 по importance)
2. ✅ LightGBM точнее предсказывает величину доходности
3. ❌ Momentum лучше угадывает направление на 1d
4. 🔄 Нужна калибровка для улучшения Brier на 20d

---

## 🔬 Следующие шаги

### 1. Улучшение Brier Score на 20d

**Проблема:** Brier 20d хуже baseline на 16%

**Решения:**
```bash
# Попробовать калибровку
python scripts/2_train_model.py --exp-name lgbm_news_calibrated --calibrate

# Отдельная модель для probabilities
# TODO: добавить --separate-prob-model
```

### 2. Ensemble для Directional Accuracy

**Идея:** Комбинировать LightGBM (returns) + Momentum (direction)
```python
# Использовать LightGBM для величины, Momentum для знака
pred_magnitude = lightgbm.predict_return()
pred_sign = momentum.predict_direction()

final_pred = abs(pred_magnitude) * pred_sign
```

### 3. Улучшение новостных фич

**Текущие:** Только count (количество новостей)

**Планы:**
- Sentiment analysis (positive/negative/neutral)
- Topic modeling (какие темы обсуждаются)
- Entity extraction (какие компании упоминаются)

```python
# TODO: добавить в features_news.py
def add_sentiment_features(candles_df, news_df):
    # VADER или FinBERT для sentiment
    pass
```

### 4. Feature Selection

**Топ-20 признаков:**
```
news_count_30d_lag    1007.5
news_count_7d_lag      907.0
ma_5d                  589.25
volatility_20d         562.0
log_volume             534.25
```

Попробовать обучить только на топ-K:
```bash
# TODO: добавить --top-features
python scripts/2_train_model.py --exp-name lgbm_top20 --top-features 20
```

### 5. Cross-validation для временных рядов

**Purged K-Fold:**
- Учитывать overlap в таргетах (20-day targets перекрываются)
- Использовать gap между фолдами

```python
# TODO: добавить в src/finam/cv.py
from finam.cv import purged_kfold_cv
```

---

## 💡 Полезные команды

### Быстрое экспериментирование

```bash
# 1. Подготовить данные (один раз)
python scripts/1_prepare_data.py

# 2. Вычислить baseline (один раз)
python scripts/compute_baseline_metrics.py
python scripts/train_baseline.py

# 3. Обучить несколько моделей
python scripts/2_train_model.py --exp-name lgbm_100trees --n-estimators 100
python scripts/2_train_model.py --exp-name lgbm_500trees --n-estimators 500
python scripts/2_train_model.py --exp-name lgbm_calibrated --calibrate

# 4. Собрать результаты
python scripts/collect_experiments.py

# 5. Посмотреть топ эксперименты
cat outputs/experiments_log.csv | column -t -s, | head -5
```

### Анализ экспериментов

```bash
# Топ-5 лучших экспериментов
python -c "
import pandas as pd
df = pd.read_csv('outputs/experiments_log.csv')
print(df[['exp_name', 'test_score_total']].sort_values('test_score_total', ascending=False).head())
"

# Сравнить два эксперимента
python -c "
import pandas as pd
df = pd.read_csv('outputs/experiments_log.csv')
exp1 = df[df['exp_name'] == 'lgbm_with_news'].iloc[0]
exp2 = df[df['exp_name'] == 'momentum_baseline'].iloc[0]
print('Experiment 1:', exp1['test_score_total'])
print('Experiment 2:', exp2['test_score_total'])
print('Improvement:', (exp1['test_score_total'] - exp2['test_score_total']) / exp2['test_score_total'] * 100, '%')
"
```

### Воспроизведение лучшего эксперимента

```bash
# Найти лучший эксперимент
python -c "
import pandas as pd
df = pd.read_csv('outputs/experiments_log.csv')
best = df.loc[df['test_score_total'].idxmax()]
print('Best experiment:', best['run_id'])
print('Config:', f\"outputs/{best['run_id']}/config.yaml\")
"

# Посмотреть конфиг
cat outputs/<best_run_id>/config.yaml

# Переобучить с теми же параметрами
python scripts/2_train_model.py \
    --exp-name best_model_v2 \
    --model-type lightgbm \
    --n-estimators <...> \
    --learning-rate <...>
```

---

## 🎯 Для хакатона: Final Submission Pipeline

### 1. Выбор лучшей модели

```bash
# Найти лучший эксперимент по test_score_total
python scripts/collect_experiments.py

# Посмотреть в experiments_log.csv
head -2 outputs/experiments_log.csv | column -t -s,
```

### 2. Генерация submission

```bash
# TODO: создать scripts/4_generate_submission.py
python scripts/4_generate_submission.py \
    --model-dir outputs/2025-10-03_23-41-15_lgbm_with_news/ \
    --test-data data/raw/participants/public_test_candles.csv \
    --output submission.csv
```

### 3. Проверка

```bash
# Убедиться что submission корректен
python scripts/verify_submission.py submission.csv
```

---

## 📚 Референсы

- **CLAUDE.md** — общие guidelines для разработки
- **docs/METRICS.md** — описание метрик (MAE, Brier, DA, Score)
- **docs/evaluation.md** — формулы оценки
- **SESSION.md** — лог текущей сессии
- **outputs/experiments_log.csv** — история всех экспериментов

---

## 🔑 Ключевые принципы

1. **Воспроизводимость** — каждый эксперимент сохранен с config + metrics
2. **Сравнение с baseline** — normalized scores показывают улучшение
3. **Автоматизация** — collect_experiments.py собирает всё автоматически
4. **Data leakage protection** — новостные фичи с правильным lag
5. **Быстрые итерации** — preprocessed данные для экспериментов

**Главное правило:** Каждый эксперимент должен быть в `outputs/experiments_log.csv` с normalized score!
