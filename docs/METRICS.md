# 📊 Метрики оценки — Подробное объяснение

> **Источник**: `docs/evaluation.md` (официальная методика оценки)

---

## 🎯 Как вычисляются таргеты (целевые переменные)

### Формулы доходности

**Таргеты** — это целевые переменные, которые мы предсказываем. Они вычисляются из **будущих цен**:

#### 1. **Доходность (Return)** — процентное изменение цены

$$
\text{target\_return\_Nd} = \frac{\text{close}_{t+N}}{\text{close}_t} - 1
$$

где:
- $\text{close}_t$ — цена закрытия в день $t$ (текущий день)
- $\text{close}_{t+N}$ — цена закрытия через $N$ торговых дней
- $N \in \{1, 20\}$ — горизонт прогноза (1 день или 20 дней)

**Пример:**
```python
# День t: close = 81.70
# День t+1: close = 82.10

target_return_1d = (82.10 / 81.70) - 1 = 0.004896 ≈ +0.49%
```

#### 2. **Направление (Direction)** — знак доходности

$$
\text{target\_direction\_Nd} = \begin{cases}
1, & \text{если } \text{target\_return\_Nd} > 0 \text{ (рост)} \\
0, & \text{если } \text{target\_return\_Nd} \leq 0 \text{ (падение или без изменений)}
\end{cases}
$$

**Пример:**
```python
target_return_1d = 0.004896  # Рост → direction = 1
target_return_20d = -0.010962  # Падение → direction = 0
```

---

### Типы доходности

**1. Накопленная доходность (Cumulative Return)** — используется в задаче

$$
R_{\text{cum}} = \frac{P_{t+N}}{P_t} - 1
$$

- Показывает **общий % изменения** за период
- **НЕ зависит** от промежуточных значений
- Пример: если цена выросла с 100 до 120, то $R = 0.20 = +20\%$

**2. Среднедневная доходность (Average Daily Return)** — НЕ используется

$$
R_{\text{avg}} = \frac{1}{N} \sum_{i=1}^{N} r_i, \quad \text{где } r_i = \frac{P_i - P_{i-1}}{P_{i-1}}
$$

- Показывает **среднее дневное изменение**
- Требует знания всех промежуточных цен

**В задаче используется накопленная доходность!**

---

### Важные особенности

#### ⚠️ Торговые vs календарные дни

- **Торговые дни** — дни когда биржа работает (пн-пт, исключая праздники)
- **20 торговых дней ≈ 1 месяц** (не 20 календарных дней!)

```python
# Пример: если 20 торговых дней, это может быть:
# 2025-04-01 → 2025-05-01 (примерно 28-30 календарных дней)
```

#### ⚠️ Последние строки не имеют таргетов

Для каждого тикера:
- **Последняя строка** → нет `target_return_1d` (нет дня $t+1$)
- **Последние 20 строк** → нет `target_return_20d` (нет дня $t+20$)

```python
# В train эти строки имеют NaN или удалены
df['target_return_20d'].isna().sum()  # > 0 для последних 20 дней каждого тикера
```

#### ✅ Проверка корректности формул

Мы проверили что организаторы использовали именно эти формулы:

```bash
python scripts/verify_targets.py

# Результат:
# ✅ Все таргеты вычислены корректно (расхождение < 1e-06)
# Макс. расхождение: 9.99e-17 (погрешность округления float64)
```

---

### Связь с предсказаниями

Наши модели должны предсказать 4 величины:

| Таргет | Тип | Формат | Что предсказываем |
|--------|-----|--------|-------------------|
| `pred_return_1d` | regression | float | Доходность за 1 день |
| `pred_return_20d` | regression | float | Доходность за 20 дней |
| `pred_prob_up_1d` | probability | [0, 1] | Вероятность роста за 1 день |
| `pred_prob_up_20d` | probability | [0, 1] | Вероятность роста за 20 дней |

**Примечание:** `pred_prob_up` — это НЕ то же самое что `target_direction`!
- `target_direction` ∈ {0, 1} — бинарный факт (выросло или нет)
- `pred_prob_up` ∈ [0, 1] — вероятность роста (может быть 0.7 = "скорее вырастет")

---

## 🎯 Итоговая метрика соревнования

```
Score = 0.7 × MAE_norm + 0.3 × Brier_norm + 0.1 × DA
```

**Финальная оценка**: среднее по двум горизонтам (1d и 20d)

```
LeaderboardScore = (Score_1d + Score_20d) / 2
```

---

## 📐 Компоненты метрики

### 1. **MAE (Mean Absolute Error)** — Точность прогноза доходности

$$
\mathrm{MAE} = \frac{1}{M} \sum_{i=1}^{M} \left| y_i - \hat{y}_i \right|
$$

**Что измеряет:** Средняя абсолютная ошибка прогноза доходности

**Единицы:** Те же что и доходность (например, если доходность = 0.05 = 5%, то MAE может быть 0.02 = 2%)

**Интерпретация:**
- **MAE = 0.02** → в среднем ошибаемся на ±2% в прогнозе доходности
- **Меньше = лучше** (0 = идеальное предсказание)

**Пример:**
```python
y_true = [0.01, -0.02, 0.03]  # Реальные доходности: +1%, -2%, +3%
y_pred = [0.015, -0.01, 0.025]  # Прогнозы: +1.5%, -1%, +2.5%

MAE = (|0.01 - 0.015| + |-0.02 - (-0.01)| + |0.03 - 0.025|) / 3
    = (0.005 + 0.01 + 0.005) / 3
    = 0.00667 ≈ 0.67%
```

**Почему MAE, а не MSE?**
- Финансовые доходности имеют **"толстые хвосты"** (fat tails)
- **MSE** сильно штрафует большие ошибки → переобучается на outliers
- **MAE** робастнее к выбросам → лучше обобщается

---

### 2. **Brier Score** — Калибровка вероятностей

$$
\mathrm{Brier} = \frac{1}{M} \sum_{i=1}^{M} \left( \mathbf{1}[y_i>0] - p_{\uparrow,i} \right)^2
$$

**Что измеряет:** Насколько хорошо калибрована вероятность роста

**Интерпретация:**
- **Brier = 0** → идеальная калибровка
- **Brier = 0.25** → плохая калибровка (как у random classifier)
- **Меньше = лучше**

**Калибровка** = "если модель говорит 'вероятность роста 80%', то действительно ли в 80% случаев был рост?"

**Пример:**
```python
y_true = [0.01, -0.02, 0.03, 0.05]  # Реальные доходности
# Направления: [рост, падение, рост, рост] = [1, 0, 1, 1]

p_up = [0.6, 0.3, 0.7, 0.8]  # Предсказанные вероятности роста

Brier = ((1 - 0.6)² + (0 - 0.3)² + (1 - 0.7)² + (1 - 0.8)²) / 4
      = (0.16 + 0.09 + 0.09 + 0.04) / 4
      = 0.095
```

**Хорошая калибровка** = надёжные вероятности, которые можно использовать для:
- Risk management (оценка вероятности потерь)
- Portfolio optimization (allocation пропорционально confidence)
- Decision making (когда ставить стоп-лосс?)

---

### 3. **DA (Directional Accuracy)** — Точность направления

$$
\mathrm{DA} = \frac{1}{M} \sum_{i=1}^{M} \mathbf{1}\!\left[\operatorname{sign}(\hat{y}_i) = \operatorname{sign}(y_i)\right]
$$

**Что измеряет:** Доля правильно угаданных направлений (знак доходности)

**Интерпретация:**
- **DA = 0.5** → случайное угадывание (как монетка)
- **DA = 0.6** → угадываем 60% направлений (хорошо!)
- **DA = 0.7+** → очень хорошо (редко достижимо на практике)
- **Больше = лучше** (1.0 = всегда угадываем знак)

**Пример:**
```python
y_true = [0.01, -0.02, 0.03, -0.01]  # Знаки: [+, -, +, -]
y_pred = [0.015, -0.01, 0.005, 0.01]  # Знаки: [+, -, +, +]
#                                       Совпадают: ✓  ✓  ✓  ✗

DA = 3 / 4 = 0.75 = 75%
```

**Почему важна DA?**
- Для трейдинга важнее **направление** чем точная величина
- Если знаем что актив вырастет, можем купить (даже если не знаем насколько)
- DA > 50% → есть предсказательная сила!

---

### 4. **Нормализация относительно Baseline**

$$
MAE_{\mathrm{norm}} = 1 - \frac{\mathrm{MAE}}{\mathrm{MAE}_{\mathrm{base}}}, \qquad Brier_{\mathrm{norm}} = 1 - \frac{\mathrm{Brier}}{\mathrm{Brier}_{\mathrm{base}}}
$$

**Что измеряет:** Насколько вы лучше простого Momentum Baseline

**Интерпретация:**
- **MAE_norm = 0** → такой же как baseline
- **MAE_norm > 0** → лучше baseline (желаемый результат!)
- **MAE_norm < 0** → хуже baseline (нужно улучшать)

**Пример:**
```python
# Ваша модель:
MAE_model = 0.016846
Brier_model = 0.263079

# Baseline (Momentum):
MAE_baseline = 0.020005
Brier_baseline = 0.262271

# Нормализация:
MAE_norm = 1 - (0.016846 / 0.020005) = 1 - 0.842 = 0.158 = +15.8% улучшение
Brier_norm = 1 - (0.263079 / 0.262271) = 1 - 1.003 = -0.003 = -0.3% ухудшение
```

---

## 🧮 Комбинированный Score

```
Score = 0.7 × MAE_norm + 0.3 × Brier_norm + 0.1 × DA
```

**Веса:**
- **70%** — точность доходности (MAE) — самое важное!
- **30%** — калибровка вероятностей (Brier) — важно для risk management
- **10%** — точность направления (DA) — бонус за правильный знак

**Почему такие веса?**
- **MAE** — основная метрика (magnitude of returns matters most)
- **Brier** — важна для калибровки (надёжные probabilities = better decisions)
- **DA** — приятный бонус (направление важно, но magnitude важнее)

**Пример расчёта:**
```python
# Из нашего эксперимента (1d):
MAE_norm = 0.158243
Brier_norm = -0.003076  # Чуть хуже baseline
DA = 0.4864

Score_1d = 0.7 × 0.158243 + 0.3 × (-0.003076) + 0.1 × 0.4864
         = 0.110770 + (-0.000923) + 0.048640
         = 0.158487
```

---

## ❓ FAQ: Почему LightGBM использует только MAE?

### Вопрос
> В метрике соревнования комбинация (MAE + Brier + DA), почему LightGBM обучается только на MAE?

### Ответ

**Мы обучаем 4 ОТДЕЛЬНЫХ модели для 4 таргетов:**

```python
# 1. Доходность 1d → regression, MAE loss
model_return_1d = LGBMRegressor(objective='mae')

# 2. Доходность 20d → regression, MAE loss
model_return_20d = LGBMRegressor(objective='mae')

# 3. Вероятность роста 1d → classification, binary loss
model_prob_up_1d = LGBMClassifier(objective='binary')

# 4. Вероятность роста 20d → classification, binary loss
model_prob_up_20d = LGBMClassifier(objective='binary')
```

### Mapping: Model → Metric

| Model | Objective | Optimizes | Evaluated with |
|-------|-----------|-----------|----------------|
| `model_return_1d` | `mae` | Minimize MAE | **MAE_1d** |
| `model_return_20d` | `mae` | Minimize MAE | **MAE_20d** |
| `model_prob_up_1d` | `binary` | Maximize log-likelihood | **Brier_1d**, **DA_1d** |
| `model_prob_up_20d` | `binary` | Maximize log-likelihood | **Brier_20d**, **DA_20d** |

### Почему не одна модель с комбинированной loss?

**1. Разные типы таргетов:**
- Return → regression (continuous)
- Probability → classification (binary)
- Direction → derived from return prediction (sign)

**2. Разные шкалы:**
- MAE ∈ [0, +∞] (доходности: ±20%)
- Brier ∈ [0, 1] (вероятности: 0-100%)
- DA ∈ [0, 1] (accuracy)

**3. Модульность:**
- Можно независимо улучшать каждую модель
- Можно экспериментировать с разными моделями для разных таргетов
- Можно использовать ensemble/stacking

### Комбинированная метрика = Evaluation, не Training

**Score** используется **ПОСЛЕ обучения** для сравнения моделей:

```python
# ПОСЛЕ обучения всех 4 моделей:
preds = model.predict(X_test)

# Evaluation:
metrics = evaluate_predictions(
    y_true_1d, y_true_20d,
    preds['pred_return_1d'], preds['pred_return_20d'],
    preds['pred_prob_up_1d'], preds['pred_prob_up_20d']
)

# Комбинированный score для сравнения моделей:
score = 0.7 * metrics['mae_norm'] + 0.3 * metrics['brier_norm'] + 0.1 * metrics['da']
```

**Это метрика для ranking/leaderboard, не loss function для gradient descent!**

---

## 🔬 Практические советы

### Улучшение MAE:
- Добавить более сложные features (cross-sectional, interactions)
- Ensemble несколько моделей
- Tune hyperparameters (learning_rate, max_depth)

### Улучшение Brier:
- **Calibration!** Использовать `CalibratedClassifierCV`
- Platt scaling для post-hoc калибровки
- Отдельная модель для probabilities (не через sigmoid от returns)

### Улучшение DA:
- Использовать Momentum baseline для направлений (у него DA > 50%)
- Ensemble: LightGBM для magnitude, Momentum для direction
- Assymetric loss functions (больше штраф за неправильное направление)

---

## 📚 Ссылки

- **MAE**: https://en.wikipedia.org/wiki/Mean_absolute_error
- **Brier Score**: https://en.wikipedia.org/wiki/Brier_score
- **Directional Accuracy**: https://en.wikipedia.org/wiki/Confusion_matrix#Definition
- **Calibration**: https://scikit-learn.org/stable/modules/calibration.html

---

## 💡 Takeaways

1. ✅ **3 метрики, 4 модели** — каждая модель оптимизирует свой таргет
2. ✅ **MAE = accuracy**, **Brier = calibration**, **DA = direction**
3. ✅ **Score = evaluation metric**, не training loss
4. ✅ **Нормализация vs baseline** показывает улучшение
5. ✅ **70% веса на MAE** → точность доходности самая важная
