# TODO - FINAM FORECAST

## Выполненные задачи

### ✅ Формат данных (2025-10-04)
- [x] Заменить все .parquet на .csv (аккуратно с массивами)
  - ✅ Проверено: массивов в колонках НЕТ
  - ✅ Изменены все скрипты: 1_prepare_data.py, 2_train_model.py, 3_evaluate.py, 4_generate_submission.py, compute_baseline_metrics.py, train_baseline.py
  - ✅ Конвертированы существующие файлы (5 файлов: train, val, test, public_test, private_test)
  - ✅ Протестирован pipeline - работает корректно
  - ✅ Удалены старые .parquet файлы
  - ✅ Обновлен README.md

---

## Текущие задачи

### 📊 Улучшение моделей
- [ ] Попробовать калибровку для улучшения Brier score на 20d
- [ ] Ensemble: LightGBM (returns) + Momentum (direction)
- [ ] Feature selection: обучить на топ-20 признаках

### 📰 Новостные фичи
- [ ] Sentiment analysis (VADER/FinBERT)
- [ ] Topic modeling (LDA/BERTopic)
- [ ] Entity extraction (упоминания компаний)

### 🎯 Submission
- [ ] Сгенерировать submission для лучшей модели
- [ ] Проверить формат submission файлов
- [ ] Загрузить на платформу

---

*Этот файл gitignored - для живой работы*
