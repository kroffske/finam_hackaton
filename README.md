# ML Pipeline для FORECAST хакатона

## 🎯 Текущий статус

**✅ Готово к использованию!**
- ✅ Данные подготовлены (train/val/test + public/private)
- ✅ Baseline метрики вычислены
- ✅ Первая модель обучена (LightGBM с новостными фичами)
- ✅ 43 признака (технические индикаторы + новости)

**📊 Последняя модель:**
- Название: `lgbm_with_news`
- Timestamp: `2025-10-04_03-56-20`
- Test MAE 1d: 0.01725 (vs baseline 0.02073)
- Test MAE 20d: 0.08780 (vs baseline 0.09607)

**🚀 Следующие шаги:**

```bash
# Собрать результаты экспериментов
python scripts/collect_experiments.py

# Посмотреть leaderboard
python scripts/show_leaderboard.py

# Сгенерировать submission для последней модели
python scripts/4_generate_submission.py --run-id 2025-10-04_03-56-20_lgbm_with_news

# Обучить новую модель с другими параметрами
python scripts/2_train_model.py --exp-name lgbm_tuned --n-estimators 1000 --learning-rate 0.01
```

---

## INSTALL

```bash
python -m nltk.downloader popular
```

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
│   ├── preprocessed/              # обработанные данные (CSV формат)
│   │   ├── train.csv              # train split с признаками
│   │   ├── val.csv                # validation split
│   │   ├── test.csv               # test split
│   │   ├── public_test.csv        # preprocessed public test
│   │   ├── private_test.csv       # preprocessed private test
│   │   └── metadata.json          # метаинформация
│   └── baseline_metrics.json      # эталонные метрики Momentum
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
│   ├── 0_news_preprocess.py       # обработка новостей и привязка к тикерам
│   ├── 0_openrouter_news_classification.py # классификация новостей через LLM
│   ├── analyze_news_tickers.py    # анализ покрытия новостей по тикерам
│   ├── 1_prepare_data.py          # подготовка + split + news + public/private
│   ├── 2_train_model.py           # обучение с train/val/test оценкой
│   ├── 3_evaluate.py              # оценка модели
│   ├── 4_generate_submission.py   # генерация submission файлов
│   ├── train_baseline.py          # обучение Momentum baseline
│   ├── compute_baseline_metrics.py # вычисление baseline
│   ├── collect_experiments.py     # сбор всех экспериментов
│   └── show_leaderboard.py        # отображение leaderboard экспериментов
│
└── src/finam/
    ├── features.py                # technical indicators
    ├── features_news.py           # NEW: news features
    ├── model.py                   # модели (LightGBM, Momentum) - только regression
    ├── metrics.py                 # метрики (MAE)
    ├── evaluate.py                # сравнение моделей
    └── cv.py                      # NEW: cross-validation для временных рядов
```

## 🚀 Quick Start Commands

```bash
# 1. Подготовить данные (один раз)
python scripts/1_prepare_data.py

# 2. Вычислить baseline метрики (один раз)
python scripts/compute_baseline_metrics.py

# 3. Обучить baseline модель для сравнения
python scripts/train_baseline.py --exp-name momentum_baseline

# 4. Обучить LightGBM модель
python scripts/2_train_model.py --exp-name lgbm_with_news --model-type lightgbm

# 5. Собрать все эксперименты и вычислить метрики
python scripts/collect_experiments.py

# 6. Показать leaderboard экспериментов
python scripts/show_leaderboard.py

# 7. Сгенерировать submission для лучшей модели
python scripts/4_generate_submission.py --run-id <timestamp>_<exp_name>

# 8. (Опционально) Оценить конкретную модель
python scripts/3_evaluate.py --exp-dir <run_id> --data test --save-report
```

---

## 📊 Workflow Details

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

### 4. Метрика оценки

**Основная метрика:**
```python
MAE (Mean Absolute Error) - средняя абсолютная ошибка прогноза доходности
```

**Расчёт:**
- `mae_1d` = MAE для 1-дневного горизонта
- `mae_20d` = MAE для 20-дневного горизонта
- `mae_mean` = среднее (mae_1d + mae_20d) / 2

**Интерпретация:**
- **Меньше = лучше** (в отличие от normalized score)
- Цель: минимизировать MAE на обоих горизонтах

---

## 📈 Результаты экспериментов

### Сравнение моделей (TEST)

| Эксперимент | Model | MAE 1d | MAE 20d | MAE mean |
|------------|-------|---------|---------|----------|
| **lgbm_with_news** | LightGBM | 0.0173 | 0.0898 | **0.0536** 🥇 |
| momentum_baseline | Momentum | 0.0207 | 0.0961 | 0.0584 |

### Улучшение vs Baseline

**LightGBM с новостями:**
- ✅ **MAE 1d**: +16.5% лучше (0.0207 → 0.0173)
- ✅ **MAE 20d**: +6.6% лучше (0.0961 → 0.0898)
- ✅ **MAE mean**: +8.2% лучше (0.0584 → 0.0536)

**Выводы:**
1. ✅ Новостные фичи ОЧЕНЬ важны (топ-2 по importance)
2. ✅ LightGBM точнее предсказывает величину доходности на обоих горизонтах
3. ✅ Улучшение стабильно на train/val/test

---

## 🔬 Следующие шаги

### 1. Улучшение новостных фич

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

### 2. Feature Selection

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

### 3. Cross-validation для временных рядов ✅ РЕАЛИЗОВАНО

**Rolling Window CV с gap:**
- Gap = 21 торговый день (защита от data leakage для t+20)
- Test size = 60 дней (3 месяца торговли)
- 5 фолдов для надежной оценки

```python
from finam.cv import rolling_window_cv, evaluate_with_cv
from finam.model import LightGBMModel

# Быстрая оценка модели с CV
model = LightGBMModel()
cv_results = evaluate_with_cv(
    model, train_df, feature_cols,
    n_splits=5, test_size=60, gap=21
)

print(f"Mean MAE 1d: {np.mean(cv_results['mae_1d']):.4f}")
print(f"Std MAE 1d:  {np.std(cv_results['mae_1d']):.4f}")
```

**📚 Документация:** [docs/cross_validation.md](docs/cross_validation.md)

---

## 💡 Полезные команды

### Быстрый старт (для первого запуска)

```bash
# 1. Подготовить данные (один раз)
python scripts/1_prepare_data.py

# 2. Вычислить baseline (один раз)
python scripts/compute_baseline_metrics.py

# 3. Обучить первую модель
python scripts/2_train_model.py --exp-name lgbm_with_news --model-type lightgbm

# 4. Собрать результаты
python scripts/collect_experiments.py

# 5. Посмотреть leaderboard
python scripts/show_leaderboard.py
```

### Быстрое экспериментирование

```bash
# 1. Подготовить данные (один раз) - УЖЕ СДЕЛАНО ✓
python scripts/1_prepare_data.py

# 2. Вычислить baseline (один раз) - УЖЕ СДЕЛАНО ✓
python scripts/compute_baseline_metrics.py
python scripts/train_baseline.py

# 3. Обучить несколько моделей
python scripts/2_train_model.py --exp-name lgbm_100trees --n-estimators 100
python scripts/2_train_model.py --exp-name lgbm_500trees --n-estimators 500
python scripts/2_train_model.py --exp-name lgbm_calibrated --calibrate

# 4. Собрать результаты
python scripts/collect_experiments.py

# 5. Посмотреть топ эксперименты
python scripts/show_leaderboard.py --top 5
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
- **docs/cross_validation.md** — кроссвалидация для временных рядов
- **docs/evaluation.md** — формулы оценки
- **SESSION.md** — лог текущей сессии
- **outputs/experiments_log.csv** — история всех экспериментов

---

## 🔑 Ключевые принципы

1. **Воспроизводимость** — каждый эксперимент сохранен с config + metrics
2. **Сравнение с baseline** — normalized scores показывают улучшение
3. **Автоматизация** — collect_experiments.py собирает всё автоматически
4. **Data leakage protection** — новостные фичи с правильным lag
5. **Быстрые итерации** — preprocessed данные (CSV формат) для экспериментов
6. **CSV формат** — совместимость, читаемость, простота обмена

**Главное правило:** Каждый эксперимент должен быть в `outputs/experiments_log.csv` с normalized score!
