# 🔍 Рефлексия по EDA Анализу — 01_candles_eda.ipynb

Дата: 2025-10-03

---

## 📊 Что было проанализировано

### ✅ Сделано хорошо:

1. **Структурированный подход** — чёткие секции от загрузки до baseline validation
2. **Baseline validation** — проверка momentum стратегии с расчётом метрик (MAE, Brier, DA)
3. **Feature engineering** — candlestick patterns, technical indicators (momentum, volatility, MA, RSI, MACD, Bollinger Bands)
4. **Train-test comparison** — статистические тесты на data drift (KS test, t-test)
5. **Визуализации** — распределения, корреляции, временные паттерны

---

## 🔧 Как можно улучшить анализ

### 1. **Временная структура**
- ➕ **Добавить**: анализ автокорреляции доходностей (ACF/PACF plots)
- ➕ **Добавить**: анализ сезонности (day of week, month effects)
- ➕ **Добавить**: выявление market regimes (volatility clustering, bull/bear periods)

**Зачем**: понять насколько сильно прошлые доходности влияют на будущие, есть ли циклические паттерны

### 2. **Cross-sectional анализ**
- ⬆️ **Улучшить**: анализ cross-ticker корреляций (какие активы движутся вместе?)
- ➕ **Добавить**: sector/industry grouping если есть мета-данные
- ➕ **Добавить**: анализ dispersion (разброс доходностей между активами в один день)

**Зачем**: выявить sector momentum, создать cross-sectional features (ranks, z-scores)

### 3. **Outliers & Risk**
- ⬆️ **Углубить**: не просто выявить outliers >3σ, но проанализировать **какие тикеры** и **в какие периоды**
- ➕ **Добавить**: анализ fat tails (kurtosis по тикерам, Q-Q plots против нормального распределения)
- ➕ **Добавить**: downside risk metrics (VaR, CVaR)

**Зачем**: понять какие активы рискованнее, когда происходят экстремальные события

### 4. **Feature engineering validation**
- ➕ **Добавить**: forward-looking bias check — убедиться что все features используют только прошлые данные
- ➕ **Добавить**: feature stability analysis — как меняются корреляции признаков с таргетом во времени
- ⬆️ **Улучшить**: показать incremental value каждой группы признаков (price-only → +momentum → +volatility → +volume)

**Зачем**: предотвратить look-ahead bias, понять какие features действительно работают

### 5. **Baseline insights**
- ➕ **Добавить**: error analysis по периодам (когда baseline работает хорошо/плохо?)
- ➕ **Добавить**: scatter plots: predicted vs actual returns
- ➕ **Добавить**: residual analysis (есть ли паттерны в ошибках?)

**Зачем**: понять в каких условиях baseline fails → где можно улучшить модель

---

## ❓ Ответ на главный вопрос: Почему даны готовые таргеты?

### Вопрос
> **Почему нам даны `target_return_1d`, `target_direction_1d`, `target_return_20d`, `target_direction_20d`?**
> Я думал это вычисляемые значения.

### Ответ

**Да, это вычисляемые значения!** Организаторы заранее рассчитали их для участников.

#### 🎯 Зачем это сделали:

1. ✅ **Единообразие** — все участники используют одинаковую формулу расчёта таргетов → честное сравнение моделей
2. ✅ **Предотвращение ошибок** — участники могут ошибиться в расчёте forward returns (например, использовать look-ahead bias)
3. ✅ **Экономия времени** — не нужно тратить время хакатона на расчёт таргетов, можно сразу сосредоточиться на моделировании

#### 📐 Как их считали:

```python
# Для каждого тикера и даты t:
target_return_1d = (close[t+1] - close[t]) / close[t]
target_direction_1d = 1 if target_return_1d > 0 else 0  # 1=рост, 0=падение

target_return_20d = (close[t+20] - close[t]) / close[t]
target_direction_20d = 1 if target_return_20d > 0 else 0
```

#### ⚠️ **ВАЖНО**:
- Эти таргеты доступны **ТОЛЬКО в train**
- В **test** их **НЕТ** — вам нужно их предсказать!
- Это стандартная практика для соревнований по ML

---

## 💡 Ключевые инсайты для модели

### 1. **Data Quality: хорошо, но есть нюансы**
- ✅ Нет пропущенных значений в ключевых полях
- ⚠️ Есть временные разрывы (gaps) — учесть при rolling features
- ⚠️ Есть outliers (>3σ) — рассмотреть winsorization или RobustScaler

### 2. **Target Distribution: нормальные, но с fat tails**
- ✅ Средние доходности близки к нулю (как и должно быть)
- ✅ Направления сбалансированы (~50% up/down) → нет class imbalance
- ⚠️ Высокий kurtosis → рассмотреть MAE loss вместо MSE

### 3. **Train-Test: схожие, но есть data drift**
- ✅ Ключевые признаки (close, volume, range) имеют схожие распределения
- ⚠️ Небольшие различия в volatility → возможен data drift
- 💡 **Рекомендация**: использовать time-series CV с embargo

### 4. **Baseline: momentum работает!**
- ✅ Momentum показывает предсказательную силу (корреляция > 0)
- ✅ DA > 50% → есть сигнал в данных!
- 💡 **Можно обогнать** добавляя:
  - Более сложные технические индикаторы
  - Cross-sectional features (sector momentum, z-scores)
  - Новостной sentiment

### 5. **Feature Engineering: что работает**
- ✅ Momentum (5d, 20d) коррелирует с будущими доходностями
- ✅ Volatility (range-based) лучше чем close-to-close
- ⚠️ Volume имеет слабую корреляцию → нужны более хитрые volume features

---

## 🚀 Рекомендации для следующих шагов

### Приоритет 1: Feature Engineering
1. **Cross-sectional features** — ranks и z-scores для momentum, volatility
2. **Advanced technical indicators** — RSI, MACD, Bollinger Bands, ATR
3. **Volume features** — относительный объём, OBV, VWAP
4. **Interaction features** — momentum × volatility, volume_ratio × price_change

### Приоритет 2: Model Development
1. **LightGBM/CatBoost** с MAE loss (робастнее к outliers)
2. **Purged K-Fold CV** для предотвращения label leakage
3. **Quantile Regression** для прогнозирования интервалов
4. **Ensemble** — price-only + news + stacking

### Приоритет 3: Validation Strategy
1. **Look-ahead bias check** — каждый feature использует только прошлое
2. **Temporal stability** — проверить как features работают в разные периоды
3. **Ablation study** — измерить вклад каждой группы features

---

## 📝 Что было добавлено в ноутбук

### ✅ Изменения:

1. **Вводная ячейка** (самое начало) — объяснение данных, таргетов и метрик простыми словами на русском
2. **Секция "Выводы"** — код-ячейка с автоматическим summary по результатам EDA
3. **Секция "Комментарии к выводам"** — подробные объяснения:
   - Почему MAE лучше MSE для финансов
   - Что такое cross-sectional features
   - Как избежать look-ahead bias
   - Зачем нужен Purged K-Fold CV
   - Почему momentum работает
   - Как улучшить volume features
4. **Финальный чек-лист** для feature engineering

---

## 🎓 Выводы

Ноутбук `01_candles_eda.ipynb` предоставляет **солидную базу** для понимания данных:
- ✅ Качественный анализ распределений
- ✅ Проверка baseline
- ✅ Feature engineering фундамент

**Следующие шаги**:
1. Применить инсайты из EDA для создания продвинутых features
2. Проанализировать новостные данные (`02_news_eda.ipynb`)
3. Построить мультимодальную модель (price + news)

**Главное помнить**:
- **Look-ahead bias** — смертельная ошибка в финансовом ML
- **Cross-sectional features** — ключ к хорошему ranking
- **Purged CV** — обязательна для честной валидации

Удачи на хакатоне! 🚀
