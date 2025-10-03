# Как читать метрики

## Главная метрика: `test_score_total`

### ✅ БОЛЬШЕ = ЛУЧШЕ

```
test_score_total: 0.185 > 0.105 > 0.052
                   ↑       ↑       ↑
                 ЛУЧШЕ   ХОРОШО  BASELINE
```

## Формула

```
Score = 0.7 × MAE_norm + 0.3 × Brier_norm + 0.1 × DA

где:
  MAE_norm   = 1 - (model_MAE / baseline_MAE)
  Brier_norm = 1 - (model_Brier / baseline_Brier)
  DA         = directional_accuracy
```

## Интерпретация

### MAE (Mean Absolute Error)
- **Меньше = лучше**
- Средняя ошибка прогноза доходности

### Brier Score
- **Меньше = лучше**
- Калибровка вероятностей (хорошо = 0, плохо = 1)

### DA (Directional Accuracy)
- **Больше = лучше**
- Доля верно угаданных направлений (0.5 = случайность)

## Как использовать

### Быстрая проверка
```bash
python scripts/show_leaderboard.py
```

### Пример вывода
```
Rank            Model                Score  Improvement
*** 1       test_next_steps        0.185       +258%
*** 2       lgbm_with_news         0.105       +109%
    3      momentum_baseline       0.052         +0%
```

### Шкала оценки

| Score | Оценка |
|-------|--------|
| > 0.2 | Отлично! |
| 0.1 - 0.2 | Хорошо! |
| 0.05 - 0.1 | Неплохо |
| ≈ 0.05 | Baseline |
| < 0.05 | Плохо |

## Почему "больше = лучше"?

Из-за нормализации:

**Baseline vs себя:**
```
MAE_norm = 1 - (0.021 / 0.021) = 0.0
Score ≈ 0.05
```

**Лучшая модель:**
```
MAE_norm = 1 - (0.017 / 0.021) = 0.19  ← положительно!
Score ≈ 0.18
```

Нормализация превращает "меньше MAE = лучше" в "больше Score = лучше".

## Выбор модели для submission

```bash
# 1. Посмотреть leaderboard
python scripts/show_leaderboard.py

# 2. Выбрать модель с максимальным test_score_total
python scripts/4_generate_submission.py --run-id <best_run_id>
```

## Подробнее

- Полный список экспериментов: `outputs/experiments_log.csv`
- Сортировка по `test_score_total` (по убыванию)
- Для submission всегда выбирайте модель с максимальным score
