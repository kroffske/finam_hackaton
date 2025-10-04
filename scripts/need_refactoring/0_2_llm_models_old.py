# Тест LLM для анализа новостей с фильтрацией по датам
import json
import time
import re
import pandas as pd
from openai import OpenAI
from datetime import datetime

# ============================================
# НАСТРОЙКИ - ИЗМЕНИ ЗДЕСЬ
# ============================================

API_KEY = "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"

# Конфигурация моделей OpenRouter
MODELS = {
    "gpt-4o-mini": {
        "id": "openai/gpt-4o-mini",
        "batch_size": 20,
        "input_price": 0.15,
        "output_price": 0.60,
        "max_tokens": 4096
    },
    "deepseek-r1-distill-llama-70b": {
        "id": "deepseek/deepseek-r1-distill-llama-70b",
        "batch_size": 20,
        "input_price": 0.03,
        "output_price": 0.13,
        "max_tokens": 4096
    },
    "qwen-2.5-72b": {
        "id": "qwen/qwen-2.5-72b-instruct",
        "batch_size": 20,
        "input_price": 0.35,
        "output_price": 0.40,
        "max_tokens": 4096
    },
    "mistral-small": {
        "id": "mistralai/mistral-small",
        "batch_size": 20,
        "input_price": 0.04,
        "output_price": 0.40,
        "max_tokens": 4096
    }
}

# ====================================
# ВЫБЕРИ МОДЕЛЬ ЗДЕСЬ
# ====================================
CURRENT_MODEL = "gpt-4o-mini"

# ====================================
# ФИЛЬТР ПО ДАТАМ
# ====================================
# Установи временной интервал для обработки новостей
# Формат: "YYYY-MM-DD" или None для обработки всех новостей
START_DATE = "2025-01-01"  # Начало периода (включительно)
END_DATE = "2025-09-08"    # Конец периода (включительно)

# Если хочешь обработать ВСЕ новости, установи обе даты в None:
# START_DATE = None
# END_DATE = None

# ====================================
# ПУТЬ К ФАЙЛУ С НОВОСТЯМИ
# ====================================
INPUT_FILE = "data/preprocessed_news/news_2_with_tickers.csv"

# ============================================

MODEL_CONFIG = MODELS[CURRENT_MODEL]
MODEL = MODEL_CONFIG["id"]
BATCH_SIZE = MODEL_CONFIG["batch_size"]
MAX_TOKENS = MODEL_CONFIG["max_tokens"]

# Инициализация клиента
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)

# Промпт для анализа
SYSTEM_PROMPT = """Ты финансовый аналитик. Анализируй новости и определяй их влияние на акции компании.

Для каждой новости определи:
1. **sentiment** (тональность):
   - **-1**: Негативная новость (падение выручки, санкции, убытки, судебные иски, негативные прогнозы)
   - **0**: Нейтральная новость (общая информация без явного влияния на акции)
   - **1**: Позитивная новость (рост прибыли, новые контракты, позитивные прогнозы, дивиденды)

2. **confidence** (уверенность от 0 до 10):
   - **0-3**: Слабое влияние (упоминание в общем контексте, незначительные новости)
   - **4-6**: Умеренное влияние (квартальные отчеты, обычные сделки)
   - **7-10**: Сильное влияние (крупные контракты, санкции, смена руководства, резкие изменения финансовых показателей)

Верни результат СТРОГО в формате JSON массива БЕЗ ПОЯСНЕНИЙ:
[
  {"index": 0, "sentiment": -1, "confidence": 8},
  {"index": 1, "sentiment": 0, "confidence": 3},
  {"index": 2, "sentiment": 1, "confidence": 6}
]"""

def analyze_batch(batch_df):
    """Анализирует батч новостей через API"""

    # Формируем промпт с ЛОКАЛЬНЫМИ индексами
    prompt = "Проанализируй следующие новости:\n\n"
    for local_idx, (orig_idx, row) in enumerate(batch_df.iterrows()):
        prompt += f"**Index {local_idx}** | Ticker: {row['tickers']}\n"
        prompt += f"Заголовок: {row['title']}\n"
        prompt += f"Текст: {row['publication'][:500]}...\n\n"

    prompt += f"\nВерни ТОЛЬКО JSON массив БЕЗ ПОЯСНЕНИЙ с sentiment и confidence для каждого index (0-{len(batch_df)-1})."

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.3
        )

        result = response.choices[0].message.content

        # Парсим JSON
        try:
            # Удаляем markdown блоки
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            # Ищем JSON массив через regex
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                json_str = json_match.group(0)
                sentiments = json.loads(json_str)
            else:
                sentiments = json.loads(result)

            # Проверяем длину результата
            if len(sentiments) != len(batch_df):
                print(f"  ⚠️ Получено {len(sentiments)} результатов вместо {len(batch_df)}")
                # Дополняем пустыми значениями
                while len(sentiments) < len(batch_df):
                    sentiments.append({"sentiment": None, "confidence": None})

            usage = response.usage
            return sentiments, usage

        except json.JSONDecodeError as e:
            print(f"  ❌ Ошибка парсинга JSON: {e}")
            print(f"  Ответ модели: {result[:200]}...")
            return [{"sentiment": None, "confidence": None}] * len(batch_df), None

    except Exception as e:
        print(f"  ❌ Ошибка API: {e}")
        return [{"sentiment": None, "confidence": None}] * len(batch_df), None

def calculate_cost(input_tokens, output_tokens):
    """Вычисляет стоимость для текущей модели"""
    input_cost = (input_tokens / 1_000_000) * MODEL_CONFIG["input_price"]
    output_cost = (output_tokens / 1_000_000) * MODEL_CONFIG["output_price"]
    return input_cost, output_cost, input_cost + output_cost

# ============================================
# ЗАГРУЗКА ДАННЫХ
# ============================================

try:
    news_df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"❌ Файл {INPUT_FILE} не найден!")
    print("Укажи правильный путь к файлу в переменной INPUT_FILE")
    exit(1)

# Преобразуем колонку с датами в datetime
news_df['publish_date'] = pd.to_datetime(news_df['publish_date'])

# Сохраняем исходное количество новостей
original_count = len(news_df)

# Фильтруем по датам
if START_DATE is not None or END_DATE is not None:
    if START_DATE is not None:
        start_dt = pd.to_datetime(START_DATE)
        news_df = news_df[news_df['publish_date'] >= start_dt]

    if END_DATE is not None:
        end_dt = pd.to_datetime(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # До конца дня
        news_df = news_df[news_df['publish_date'] <= end_dt]

    if START_DATE and END_DATE:
        date_range_str = f"{START_DATE} - {END_DATE}"
    elif START_DATE:
        date_range_str = f"с {START_DATE}"
    else:
        date_range_str = f"до {END_DATE}"
else:
    date_range_str = "все даты"

# Сбрасываем индексы после фильтрации
news_df = news_df.reset_index(drop=True)

if len(news_df) == 0:
    print(f"❌ После фильтрации по датам не осталось новостей!")
    exit(1)

print(f"\n{'='*60}")
print(f"🚀 ЗАПУСК АНАЛИЗА")
print(f"{'='*60}")
print(f"Модель: {CURRENT_MODEL} ({MODEL})")
print(f"Batch size: {BATCH_SIZE}")
print(f"📅 Период: {date_range_str}")
print(f"📊 Отфильтровано: {len(news_df)} из {original_count} новостей")
print(f"Примерное количество запросов: {(len(news_df)-1)//BATCH_SIZE + 1}")
print(f"\n💰 ЦЕНЫ МОДЕЛИ:")
print(f"  Input:  ${MODEL_CONFIG['input_price']:.3f} / 1M токенов")
print(f"  Output: ${MODEL_CONFIG['output_price']:.3f} / 1M токенов")
print(f"{'='*60}\n")

# Инициализируем колонки
news_df['sentiment'] = None
news_df['confidence'] = None

# Обрабатываем батчами
total_input_tokens = 0
total_output_tokens = 0
start_time = time.time()

for i in range(0, len(news_df), BATCH_SIZE):
    batch = news_df.iloc[i:i+BATCH_SIZE]
    batch_num = i//BATCH_SIZE + 1
    total_batches = (len(news_df)-1)//BATCH_SIZE + 1

    print(f"Батч {batch_num}/{total_batches} ({i}-{min(i+BATCH_SIZE, len(news_df))})...", end=" ")

    sentiments, usage = analyze_batch(batch)

    # Сохраняем результаты
    for local_idx, sent_data in enumerate(sentiments):
        global_idx = i + local_idx
        if global_idx < len(news_df):
            news_df.loc[global_idx, 'sentiment'] = sent_data.get('sentiment')
            news_df.loc[global_idx, 'confidence'] = sent_data.get('confidence')

    if usage:
        total_input_tokens += usage.prompt_tokens
        total_output_tokens += usage.completion_tokens

        # Вычисляем стоимость на текущий момент
        input_cost, output_cost, total_cost = calculate_cost(total_input_tokens, total_output_tokens)

        print(f"✓ {usage.prompt_tokens}↑/{usage.completion_tokens}↓ | Потрачено: ${total_cost:.4f}")
    else:
        print(f"⚠️ Ошибка")

    # Сохраняем промежуточный результат
    start_str = START_DATE.replace('-', '') if START_DATE else 'all'
    temp_filename = f'news_2_with_tickers_llm_{start_str}_TEMP.csv'
    news_df.to_csv(temp_filename, index=False)

    time.sleep(0.5)  # Задержка между запросами

elapsed_time = time.time() - start_time

# ============================================
# РЕЗУЛЬТАТЫ
# ============================================

print(f"\n{'='*60}")
print(f"📊 РЕЗУЛЬТАТЫ ТЕСТА")
print(f"{'='*60}")
print(f"Модель: {CURRENT_MODEL}")
print(f"Период: {date_range_str}")
print(f"Обработано новостей: {len(news_df)}")
print(f"Успешных: {news_df['sentiment'].notna().sum()}")
print(f"Ошибок: {news_df['sentiment'].isna().sum()}")
print(f"Время выполнения: {elapsed_time/60:.2f} минут")

print(f"\n📈 ИСПОЛЬЗОВАНО ТОКЕНОВ:")
print(f"  Input:  {total_input_tokens:,}")
print(f"  Output: {total_output_tokens:,}")
print(f"  TOTAL:  {total_input_tokens + total_output_tokens:,}")

# Подсчет стоимости
if total_input_tokens > 0:
    input_cost, output_cost, total_cost = calculate_cost(total_input_tokens, total_output_tokens)

    print(f"\n💰 СТОИМОСТЬ ({CURRENT_MODEL}):")
    print(f"  Input:  ${input_cost:.6f}")
    print(f"  Output: ${output_cost:.6f}")
    print(f"  TOTAL:  ${total_cost:.6f}")

    # Прогноз для полного датасета (60k новостей)
    if original_count > len(news_df):
        scale_factor = original_count / len(news_df)
        projected_input = total_input_tokens * scale_factor
        projected_output = total_output_tokens * scale_factor
        proj_in_cost, proj_out_cost, proj_total = calculate_cost(projected_input, projected_output)
        projected_time = elapsed_time * scale_factor

        print(f"\n🔮 ПРОГНОЗ ДЛЯ ВСЕХ {original_count:,} НОВОСТЕЙ:")
        print(f"  Токены: {projected_input:,.0f} input / {projected_output:,.0f} output")
        print(f"  Стоимость: ${proj_total:.2f}")
        print(f"  Время: ~{projected_time/60:.1f} минут")

# Статистика
valid_sentiments = news_df[news_df['sentiment'].notna()]
if len(valid_sentiments) > 0:
    print(f"\n📋 РАСПРЕДЕЛЕНИЕ SENTIMENT:")
    sentiment_counts = valid_sentiments['sentiment'].value_counts().sort_index()
    for sent, count in sentiment_counts.items():
        label = {-1: "📉 Негатив", 0: "➖ Нейтрал", 1: "📈 Позитив"}.get(sent, "❓")
        print(f"  {label}: {count} ({count/len(valid_sentiments)*100:.1f}%)")

    print(f"\n📊 Средняя уверенность: {valid_sentiments['confidence'].mean():.2f}/10")

    # Примеры
    print(f"\n📰 ПРИМЕРЫ РЕЗУЛЬТАТОВ:")
    for idx, row in valid_sentiments.head(3).iterrows():
        sent_label = {1: "📈 ПОЗИТИВ", -1: "📉 НЕГАТИВ", 0: "➖ НЕЙТРАЛ"}.get(row['sentiment'], "❓")
        print(f"\n{sent_label} (уверенность: {row['confidence']}/10) | {row['tickers']}")
        print(f"  {row['title'][:80]}...")

# Финальное сохранение
start_str = START_DATE.replace('-', '') if START_DATE else 'all'
output_filename = f'news_2_with_tickers_llm_{start_str}.csv'
news_df.to_csv(output_filename, index=False)
print(f"\n💾 Результаты сохранены в {output_filename}")
print(f"{'='*60}")
