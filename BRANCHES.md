# Git Workflow

## Правило одно: main всегда работает

## Ветки
| Тип | Формат | Пример |
|---|---|---|
| Новый модуль | feature/название | feature/deduplication-step |
| Баг | fix/описание | fix/parser-keyerror |
| Промпт | experiment/что | experiment/evidence-fewshot |
| RAG | experiment/что | experiment/norms-indexing |

## Workflow для каждой задачи

### 1. Начать задачу
```bash
git checkout dev
git checkout -b feature/название-задачи
```

### 2. Работать и коммитить часто
```bash
git add файл.py
git commit -m "тип: короткое описание"
```

### 3. Типы коммитов
| Префикс | Когда |
|---|---|
| feat: | новая функциональность |
| fix: | исправление бага |
| prompt: | изменение промпта |
| config: | изменение порогов/весов |
| test: | добавление тестов |
| chore: | служебные изменения |
| refactor: | рефакторинг без изменения поведения |

### 4. Влить в dev
```bash
git checkout dev
git merge feature/название-задачи
```

### 5. Влить в main (только когда протестировано)
```bash
git checkout main
git merge dev
git tag v0.X.0
```

## Правила для experiment/ веток
- Никогда не мержить в main напрямую
- Сначала в dev, запустить pytest, потом в main
- Промпт-эксперименты можно удалять после завершения:
  git branch -d experiment/название