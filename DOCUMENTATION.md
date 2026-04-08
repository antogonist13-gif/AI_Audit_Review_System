# AI-assisted Audit Review System — Техническая документация (v2.0)

**Дата:** 08.04.2026 | **Статус:** Рабочий прототип

---

## Назначение

Система автоматически обрабатывает российские аудиторские акты (DOCX/PDF): извлекает нарушения, оценивает их по трём осям качества, улучшает юридические формулировки и генерирует итоговый чеклист с рекомендациями.

---

## Технологический стек

| Компонент | Технология |
|---|---|
| Язык | Python 3.14 |
| LLM (локальный) | Ollama — `gemma3:12b` (по умолчанию), поддерживаются `llama3:8b`, `mistral:7b`, `qwen2.5:7b` |
| Эмбеддинги | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Векторная БД | ChromaDB (PersistentClient, 3 коллекции) |
| Полнотекстовый поиск | BM25 (in-memory, персистированный отдельно) |
| Морфология | pymorphy2 (лемматизация русских текстов) |
| UI | Streamlit |
| Парсинг документов | python-docx, pdfplumber |

---

## Архитектура: модули и их роли

```
app.py                    ← Streamlit UI (фронтенд)
    └── act_pipeline.py   ← Главный оркестратор (единственная точка входа)
            ├── act_preprocessor.py    ← Извлечение нарушений из документа
            ├── violation_normalizer.py ← Лемматизация + ключевые слова
            ├── act_retrieval.py        ← Гибридный поиск (vector + BM25)
            ├── violation_evaluator.py  ← 3 параллельных LLM-оценки
            ├── comparison_engine.py    ← Сравнение с эталонами
            ├── formulation_improver.py ← Улучшение формулировки + grounding
            └── checklist_builder.py    ← Сборка итогового ChecklistItem
                    └── verifier.py     ← Детерминированные проверки (без LLM)

multi_indexer.py    ← Индексация справочных документов в ChromaDB
embeddings.py       ← Обёртка над sentence-transformers
rag_pipeline.py     ← Обёртка над Ollama с кэшированием
llm_cache.py        ← JSON-кэш LLM-ответов (TTL 24ч)
bm25_cache.py       ← In-memory BM25-индексы
report_generator.py ← Генерация отчётов (.docx и .md)
models.py           ← Все датаклассы
config.py           ← Все настройки и пороги
prompts.py          ← Шаблоны промптов и парсеры ответов
```

---

## Центральная модель данных

```
Violation               ← Сырое нарушение из акта
ViolationContext        ← Нарушение + все результаты retrieval
ChecklistItem           ← Финальный выход пайплайна (один на нарушение)
ImprovedFormulation     ← Результат улучшения формулировки LLM
VerificationResult      ← Результат одной детерминированной проверки
ItemTrace               ← Полная трассировка (для аудита решений)
```

---

## Основной пайплайн (`analyze_act`)

Для каждого нарушения выполняется строго последовательно:

1. **Извлечение** (`act_preprocessor`) — 3-уровневая стратегия:
   - Path 1: Чтение таблицы нарушений из DOCX
   - Path 2: Regex-сегментация (резолютивная / описательная зоны)
   - Path 3: LLM fallback (когда структура не распознана)

2. **Нормализация** (`violation_normalizer`) — лемматизация через pymorphy2, извлечение ключевых слов, флаг `possibly_not_a_violation` при < 20 токенов

3. **Retrieval** (`act_retrieval`) — **вызывается ровно один раз** на нарушение. Гибридное ранжирование: `0.5 × cosine + 0.3 × BM25 + 0.2 × law_boost` по трём коллекциям (norms, typical, historical)

4. **Оценка** (`violation_evaluator`) — 3 параллельных LLM-вызова:
   - **Доказательность** (evidence): наличие дат, сумм, документов, ФИО
   - **Правовая корректность** (legal): соответствие нормы ситуации
   - **Исполнимость** (actionability): понятность предписания

5. **Улучшение** (`formulation_improver`) — LLM переформулирует нарушение в официально-деловом стиле + deterministic grounding check (fuzzy-matching квалификации с нормативной БД)

6. **Сборка** (`checklist_builder`) — детерминированное присвоение статусов по числовым порогам, применение verifier-overrides поверх LLM-оценок

---

## Справочная база знаний

Три ChromaDB-коллекции, наполняются вручную через UI или API:

| Коллекция | Назначение | Путь |
|---|---|---|
| `norms` | Нормативные документы (законы, ФЗ, ГОСТ, СП) | `data/db/norms` |
| `typical_violations` | Каталог типовых нарушений (JSON) | `data/db/typical` |
| `historical_checklists` | Архивные акты / чеклисты | `data/db/historical` |

**Критично:** При пустых коллекциях LLM работает без контекста — качество оценок резко снижается.

---

## Пороги и веса (`config.py`)

```python
# Пороги статусов
EVIDENCE_SUFFICIENT_THRESHOLD = 0.4   # LLM-оценка < 0.4 → insufficient
LEGAL_CORRECT_THRESHOLD       = 0.5
ACTIONABILITY_THRESHOLD       = 0.5
LAW_GROUNDING_FUZZY_MIN       = 0.7   # порог fuzzy-match для grounding

# Веса confidence score
EVIDENCE_WEIGHTS = {"evidence": 0.30, "legal": 0.30, "actionability": 0.20, "similarity": 0.20}

# Веса гибридного поиска
HYBRID_SCORE_WEIGHTS = {"cosine": 0.5, "bm25": 0.3, "law_boost": 0.2}
```

---

## Запуск

```bash
# 1. Создать виртуальное окружение и установить зависимости
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Запустить Ollama с нужной моделью
ollama pull gemma3:12b && ollama serve

# 3. Запустить Streamlit UI
streamlit run app.py
# или через скрипт:
./run.sh
```

---

## Известные ограничения (текущее состояние)

| Проблема | Причина |
|---|---|
| Низкое качество LLM-оценок | Локальная модель (gemma3:12b) слабо следует структурированным промптам; нормативная база пуста |
| Grounding почти всегда False | В коллекции `norms` нет документов — fuzzy-matching не находит совпадений |
| Нет контекстного сопоставления | Парсинг контекста (`_find_context_in_descriptive`) часто промахивается, если law_ref короткий |
| Параллелизм Streamlit | Фоновые потоки используют глобальный `pipeline_stats` без блокировок |
| Нет валидации входных данных | DOCX с нестандартной структурой таблицы падает на LLM fallback без предупреждения |

---

## Структура тестов

```
tests/
  test_act_pipeline.py        ← Интеграционные тесты пайплайна
  test_act_preprocessor.py    ← Юнит-тесты извлечения нарушений
  test_act_retrieval.py        ← Тесты гибридного поиска
  test_checklist_builder.py   ← Тесты сборки и статусов
  test_violation_normalizer.py ← Тесты нормализации
  test_prompts.py              ← Тесты парсеров промптов
  test_verifier.py             ← Тесты детерминированных проверок
  fixtures/                    ← Тестовые документы
```

Запуск: `pytest tests/`

---

## Приоритеты для следующих итераций

1. **Наполнить нормативную базу** — без нормативных документов в ChromaDB система работает вслепую
2. **Улучшить промпты** — добавить few-shot примеры в шаблоны оценки, улучшить парсинг структурированных ответов
3. **Улучшить `_find_context_in_descriptive`** — сделать более точное сопоставление нарушений из таблицы с описательной частью акта
4. **Добавить более сильную модель** — переход на API (GPT-4o, Claude) или более крупную локальную модель
5. **Исправить thread safety** — `pipeline_stats` должен быть per-request, не глобальным

---

Документация описывает состояние кода на **08.04.2026**. Центральный файл для входа в кодовую базу — `act_pipeline.py`.
