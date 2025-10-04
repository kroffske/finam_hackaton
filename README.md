# ML Pipeline для FORECAST хакатона

## 🚀 Быстрый старт

# Данные

```
Заменяем исходные данные из `data/raw/participants` на файлы с такими же названиями

data/raw/participants/candles_2.csv - данные на которых предсказываем 
data/raw/participants/news_2.csv - данные на которых предсказываем 

submition будет в папке с артефактами outputs/ 
```

```bash
# 1. Создаем и активируем виртуальную среду
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# 2. Ставим Poetry и зависимости
python -m pip install poetry
poetry install --no-root           # или poetry install

# ключ запуска к openrouter должен лежать в .env OPENROUTER_API_KEY='sk-or-v....'

# 3. Запускаем полный train-пайплайн
# если нужно сходить в LLM 
poetry run python train.py --exp-name production --model-type lightgbm --start-date 2024-01-01 --force-llm --train-ratio 0.86 --val-ratio 0.07 

# запуск, если мы уже ХОДИЛИ в LLM (скипает если файл news_2_with_tickers_llm.csv - создан)
# poetry run python train.py --exp-name final_24 --model-type lightgbm --skip-llm --start-date 2024-01-01 --skip-llm  --train-ratio 0.86 --val-ratio 0.07

# 4. Генерируем сабмишн (последняя модель хранится в outputs/latest)
poetry run python inference.py --run-id latest 

-- full (можно добавить ключ - будет для всех дней )

# Опционально: если есть OPENROUTER ключ для LLM
export OPENROUTER_API_KEY="sk-..."
```





Все что нужно запускать. 
ниже описание.


























> **Примечание.** Все команды выполняются из корня репозитория. Скрипт `train.py`
> собирает пайплайн из README (0_1 → 0_2 → … → 2_train → collect), а `inference.py`
> переиспользует те же стадии для тестовых данных и вызывает генерацию submission.

## 🚀 Быстрый старт

```bash
# 0. (опционально) задать ключ для LLM-сентимента
export OPENROUTER_API_KEY='your-key-here'

# 1. Создаём виртуальное окружение и ставим Poetry
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install poetry

# 2. Устанавливаем зависимости проекта
poetry install

# 3. Запускаем полный train-пайплайн (создаст outputs/<run_id>)
poetry run python train.py --exp-name my_experiment --model-type lightgbm

# 4. Генерируем сабмишен (по умолчанию использует outputs/latest)
poetry run python inference.py --run-id latest --full
```

После успешного обучения скрипт `train.py` обновляет симлинк `outputs/latest`,
который удобно использовать для инференса и генерации submission.

## 🎯 Текущий статус

**✅ Pipeline готов!**
- ✅ Данные подготовлены (train/val/test splits)
- ✅ 57 признаков (технические + новостные counts)
- ✅ Baseline обучен (Momentum)
- ✅ LightGBM модель обучена
- ✅ Submission сгенерирован

**📊 Лучшая модель:**
- `lgbm_with_news` (2025-10-04_22-09-17)
- Val MAE mean: **0.0446** (↓2.7% vs Momentum)
- Test MAE mean: **0.0566** (↓9.0% vs 0.0622)

**🔥 Следующий шаг: LLM sentiment analysis**

```bash
# 1. Установить OpenRouter API key
export OPENROUTER_API_KEY='your-key-here'

# 2. Запустить LLM pipeline (см. раздел ниже)
python scripts/0_2_llm_models.py
python scripts/0_3_llm_explode.py --all
python scripts/0_4_news_ticker_features.py
python scripts/1_prepare_data.py
python scripts/2_train_model.py --exp-name lgbm_with_llm
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
│   │   ├── candles.csv            # train свечи
│   │   ├── news.csv               # train новости
│   │   ├── candles_2.csv          # holdout test свечи
│   │   └── news_2.csv             # holdout test новости
│   ├── preprocessed_news/         # обработанные новости
│   │   ├── news_with_tickers.csv         # новости + тикеры
│   │   ├── news_with_tickers_llm.csv     # + LLM sentiment
│   │   ├── news_ticker_sentiment.csv     # exploded по тикерам
│   │   ├── news_ticker_features.csv      # aggregated features
│   │   ├── news_2_with_tickers.csv       # test новости + тикеры
│   │   ├── news_2_with_tickers_llm.csv   # test + LLM sentiment
│   │   ├── news_2_ticker_sentiment.csv   # test exploded
│   │   └── news_2_ticker_features.csv    # test aggregated
│   ├── preprocessed/              # обработанные данные (CSV формат)
│   │   ├── train.csv              # train split с признаками
│   │   ├── val.csv                # validation split
│   │   ├── test.csv               # test split
│   │   ├── holdout_test.csv       # preprocessed holdout test
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
│   ├── 0_1_news_preprocess.py     # обработка новостей и привязка к тикерам
│   ├── 0_2_llm_models.py          # LLM sentiment analysis (OpenRouter/gpt-4o-mini)
│   ├── 0_3_llm_explode.py         # explode новостей по тикерам + news_type
│   ├── 0_4_news_ticker_features.py # агрегация ticker features (counts + sentiment)
│   ├── 1_prepare_data.py          # подготовка + split + join news features
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
    ├── features_news_tickers.py   # news features aggregation (counts + LLM sentiment)
    ├── news_tickers_v2.py         # ticker assignment logic
    ├── llm_sentiment.py           # LLM sentiment analysis via OpenRouter
    ├── model.py                   # модели (LightGBM, Momentum) - только regression
    ├── metrics.py                 # метрики (MAE)
    ├── evaluate.py                # сравнение моделей
    └── cv.py                      # cross-validation для временных рядов
```

## 🚀 Quick Start Commands

### Базовый Pipeline (✅ ВЫПОЛНЕН)

```bash
# Базовый pipeline - простые новостные counts без LLM
python scripts/0_1_news_preprocess.py           # тикеры из новостей
python scripts/0_3_llm_explode.py --all         # explode по тикерам
python scripts/0_4_news_ticker_features.py      # агрегация counts
python scripts/1_prepare_data.py                # train/val/test split
python scripts/2_train_model.py --exp-name momentum_baseline --model-type momentum
python scripts/2_train_model.py --exp-name lgbm_with_news --model-type lightgbm
python scripts/collect_experiments.py           # собрать все эксперименты
python scripts/4_generate_submission.py --run-id 2025-10-04_22-09-17_lgbm_with_news
```

**Результат:**
- 57 признаков (техн. индикаторы + news_count_1d/7d/30d)
- Val MAE: 0.0446 (LightGBM) vs 0.0459 (Momentum)
- Submission готов: `outputs/2025-10-04_22-09-17_lgbm_with_news/submission.csv`

---

### 🔥 LLM Pipeline (следующий шаг)

**Добавляет:** sentiment analysis через gpt-4o-mini (OpenRouter API)

```bash
# 0. Setup API key
# внутри скрипта есть loadenv - положить ключ в .env и все заработает 
export OPENROUTER_API_KEY='sk-or-v1-...'  # https://openrouter.ai/keys

# 1. LLM sentiment analysis
python scripts/0_2_llm_models.py
# → news_with_tickers_llm.csv (+ sentiment: -1/0/+1, confidence: 0-10)

# 2. Explode + агрегация с LLM features
python scripts/0_3_llm_explode.py --all
python scripts/0_4_news_ticker_features.py

# 3. Переобучить модель с новыми фичами
python scripts/1_prepare_data.py  # перегенерирует train/val/test с LLM фичами
python scripts/2_train_model.py --exp-name lgbm_with_llm --model-type lightgbm

# 4. Сравнить результаты
python scripts/collect_experiments.py
```

**Ожидаемые фичи:**
- `sentiment_mean` — средний sentiment по новостям
- `sentiment_weighted` — взвешенный по confidence
- `positive_count`, `negative_count`, `neutral_count`
- Rolling features: `sentiment_mean_7d`, `sentiment_mean_30d`

**Стоимость:** ~$0.01-0.05 за 1000 новостей (batch processing, 6 параллельных запросов)

---

## 📊 Workflow Details

---

## 📊 Ключевые улучшения

### 1. Новостные фичи (базовые - counts)

**Добавлены в `features_news_tickers.py`:**
- `news_count_1d` — количество новостей за день
- `news_count_7d` — за последние 7 дней (rolling)
- `news_count_30d` — за последние 30 дней (rolling)

**ВАЖНО:** Правильный lag для избежания data leakage
- Новости доступны до `t-1`
- Для свечей дня `t` используем новости до `t-1`

**Feature Importance (базовый pipeline):**
```
1. news_count_30d  — 1007.5 🥇
2. news_count_7d   — 907.0  🥈
3. ma_5d           — 589.25
```

### 2. LLM Sentiment Features 🆕

**Pipeline:** gpt-4o-mini via OpenRouter API

**Добавленные признаки:**
- `sentiment_mean` — средний sentiment по новостям (-1 до 1)
- `sentiment_weighted` — sentiment взвешенный по уверенности модели
- `confidence_mean` — средняя уверенность LLM (0-10)
- `positive_count` — количество позитивных новостей
- `negative_count` — количество негативных новостей
- `neutral_count` — количество нейтральных новостей
- Rolling features для всех метрик (7d, 30d)

**Пример:**
```python
# После агрегации на (date, ticker) уровне
ticker_features.head()
   date       ticker  news_count_1d  sentiment_mean  confidence_mean  positive_count
0  2025-04-15  SBER   5             0.6            7.2              3
1  2025-04-15  GAZP   3             -0.33          6.5              1
```

**Классификация news_type:**
- `company_specific` — одна компания (1 тикер)
- `market_wide` — общерыночные новости (ticker = 'MARKET')
- `market_wide_company` — несколько компаний (2+ тикера)

**Стоимость LLM обработки:**
- Модель: gpt-4o-mini ($0.15/1M input tokens, $0.60/1M output tokens)
- ~20 новостей за 1 API запрос (batch processing)
- 6 параллельных запросов
- Примерная стоимость: $0.01-0.05 за 1000 новостей

### 3. Автоматическая оценка на test

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

### 4. Система трекинга экспериментов

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

### 5. Метрика оценки

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

### Результаты (Val / Test)

| Модель | Features | Val MAE 1d | Val MAE 20d | Val MAE mean | Test MAE mean |
|--------|----------|------------|-------------|--------------|---------------|
| **LightGBM + news** | 57 | 0.0125 | 0.0683 | **0.0446** 🥇 | **0.0566** |
| LightGBM (43 fts) | 43 | 0.0126 | 0.0695 | 0.0458 | 0.0566 |
| Momentum baseline | 57 | 0.0151 | 0.0686 | 0.0459 | 0.0622 |

### Улучшение vs Baseline

**LightGBM + новостные counts:**
- ✅ Val MAE: 0.0446 vs 0.0459 (↓2.7%)
- ✅ Test MAE: 0.0566 vs 0.0622 (↓9.0%)
- ✅ Новостные фичи в топ-3 по feature importance

**Submission готов:**
- 19 тикеров × 20 горизонтов
- `outputs/2025-10-04_22-09-17_lgbm_with_news/submission.csv`

---

## 🔬 Следующие шаги

### 1. LLM Sentiment Analysis 🔥

**Запустить:** см. раздел "LLM Pipeline" выше

**Ожидаемые улучшения:**
- Sentiment features (positive/negative/neutral counts)
- Confidence-weighted sentiment
- News type classification (company_specific vs market_wide)

**Стоимость:** ~$0.01-0.05 за 1000 новостей

### 2. Hyperparameter Tuning

```bash
# Подобрать гиперпараметры LightGBM
python scripts/2_train_model.py --exp-name lgbm_tuned \
  --n-estimators 500 \
  --learning-rate 0.01 \
  --max-depth 7
```

### 3. Feature Engineering

**Идеи:**
- Sector momentum (агрегация по секторам)
- Market regime detection (volatility clustering)
- Technical indicators: ADX, ATR, Stochastic
- Cross-ticker correlations

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

### Быстрое экспериментирование

```bash
# Обучить модель с разными параметрами
python scripts/2_train_model.py --exp-name lgbm_500trees --n-estimators 500
python scripts/2_train_model.py --exp-name lgbm_deep --max-depth 8

# Собрать все эксперименты
python scripts/collect_experiments.py

# Посмотреть leaderboard
python -c "
import pandas as pd
df = pd.read_csv('outputs/experiments_log.csv')
print(df[['exp_name', 'val_mae_mean', 'test_mae_mean']].sort_values('val_mae_mean'))
"
```

### Анализ feature importance

```bash
# Посмотреть важность признаков лучшей модели
python -c "
import pandas as pd
fi = pd.read_csv('outputs/2025-10-04_22-09-17_lgbm_with_news/feature_importance.csv')
print(fi.head(20))
"
```

### Генерация submission

```bash
# Для лучшей модели
python scripts/4_generate_submission.py --run-id 2025-10-04_22-09-17_lgbm_with_news

# Для всех дат (не только последняя)
python scripts/4_generate_submission.py --run-id <run_id> --full
```

---

## 🎯 Final Submission

**Готово!** Submission файл сгенерирован:
- `outputs/2025-10-04_22-09-17_lgbm_with_news/submission.csv`
- 19 тикеров × 20 горизонтов (p1-p20)

**Загрузить на платформу соревнования.**

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
