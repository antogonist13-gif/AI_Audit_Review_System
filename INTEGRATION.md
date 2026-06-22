# Интеграция дизайн-системы с СВК Analytics

## ✅ Что реализовано

### 1. **FastAPI Backend** (`api_server.py`)
REST API для получения данных из Python:

**Endpoints:**
- `GET /api/overview` - Обзорные метрики с фильтрами
- `GET /api/violations` - Данные о нарушениях
- `GET /api/maturity` - Зрелость СВК
- `GET /api/directions` - Покрытие направлений
- `GET /api/form-gap` - Разрыв формы СВК
- `GET /api/risk-organizations` - Топ организаций в зоне риска
- `GET /api/data-quality` - Качество данных
- `GET /api/filters/options` - Опции для фильтров

### 2. **Next.js Pages**

#### Главная страница (`app/page.tsx`)
- ✅ 12 KPI карточек с ключевыми метриками
- ✅ Фильтры (федеральный округ, регион, тип организации, группа риска)
- ✅ Автоматическая загрузка данных из API
- ✅ Обработка ошибок и loading states

#### Страница "Зрелость СВК" (`app/maturity/page.tsx`)
- ✅ 6 интерактивных графиков (Recharts):
  - Базовые элементы СВК
  - Фактическая форма СВК
  - Отмеченные формы
  - Рискоориентированный подход
  - Оценка эффективности
  - Совершенствование СВК

#### Страница "Организации в зоне риска" (`app/risk/page.tsx`)
- ✅ Таблица с топ-100 организаций (TanStack Table)
- ✅ Сортировка по всем колонкам
- ✅ Поиск по названию организации
- ✅ Цветные badges для групп риска
- ✅ Экспорт в CSV

### 3. **Интеграция**
- ✅ Все компоненты дизайн-системы используются
- ✅ Responsive design
- ✅ Dark mode support
- ✅ Фильтры с active pills
- ✅ Loading states для всех страниц

## 🚀 Запуск полного стека

### Шаг 1: Установка зависимостей

```bash
# Python backend
cd /Users/user/svk_analytics_project
pip install fastapi uvicorn

# Frontend (уже установлено)
cd frontend
npm install
```

### Шаг 2: Запуск API сервера

```bash
# В корневой папке проекта
python api_server.py
```

API будет доступен на **http://localhost:8000**

### Шаг 3: Запуск Next.js frontend

```bash
# В папке frontend
cd frontend
npm run dev
```

Frontend будет доступен на **http://localhost:3001**

### Шаг 4: Откройте в браузере

Перейдите на http://localhost:3001 и используйте дашборд!

## 📁 Структура интеграции

```
svk_analytics_project/
├── api_server.py           # ⭐ FastAPI REST API
├── app.py                  # Streamlit (старый интерфейс)
├── frontend/               # ⭐ Next.js приложение
│   ├── app/
│   │   ├── page.tsx        # ⭐ Главная (12 KPI)
│   │   ├── maturity/
│   │   │   └── page.tsx    # ⭐ Зрелость СВК
│   │   └── risk/
│   │       └── page.tsx    # ⭐ Организации в риске
│   └── components/
│       └── dashboard/      # Дизайн-система
└── src/svk_analytics/      # Python модули анализа
```

## 🎯 Доступные страницы

1. **Обзор** (`/`) - 12 KPI карточек + фильтры
2. **Зрелость СВК** (`/maturity`) - 6 графиков
3. **Организации в риске** (`/risk`) - Таблица топ-100 + экспорт

## 🔄 Поток данных

```
Python модули → FastAPI → REST API → Next.js → React компоненты
(src/svk_analytics) (api_server.py) (fetch) (app/) (components/dashboard)
```

## 🎨 Используемые компоненты дизайн-системы

- ✅ `DashboardShell` - Layout
- ✅ `SidebarNavigation` - Навигация (обновлена под СВК)
- ✅ `TopBar` - Верхняя панель
- ✅ `KpiCard` - 12 метрик на главной
- ✅ `ChartCard` + `ChartContainer` - Графики Recharts
- ✅ `DataTable` - Таблица организаций
- ✅ `FilterBar` - Фильтры с active pills
- ✅ `LoadingCards`, `LoadingChart`, `LoadingTable` - Loading states
- ✅ `EmptyStateError` - Обработка ошибок

## 📊 Примеры интеграции

### KPI Card с данными из API

```tsx
<KpiCard
  title="Организаций"
  value={data.total_organizations}
  icon={Users}
  format="number"
  description="Всего организаций в выборке"
/>
```

### График с Recharts

```tsx
<ChartCard title="Базовые элементы СВК">
  <ChartContainer>
    <ResponsiveContainer>
      <BarChart data={data.elements}>
        <Bar dataKey="yes_orgs" fill="hsl(var(--chart-1))" />
      </BarChart>
    </ResponsiveContainer>
  </ChartContainer>
</ChartCard>
```

### Таблица с TanStack Table

```tsx
<DataTable
  columns={columns}
  data={data}
  searchKey="org_name"
  searchPlaceholder="Поиск организации..."
/>
```

## 🔧 Конфигурация

### Фильтры
Фильтры автоматически загружаются из API и применяются ко всем метрикам:
- Федеральный округ
- Регион
- Тип организации
- Группа риска

### Год анализа
По умолчанию: 2025. Можно изменить в каждом компоненте через query параметр `year`.

## 🐛 Troubleshooting

### API сервер не запускается
```bash
# Проверьте, что данные есть в data/raw/
ls data/raw/

# Установите зависимости
pip install -r requirements.txt
```

### CORS ошибки
API сервер настроен на `localhost:3000` и `localhost:3001`. Если используете другой порт, обновите `api_server.py`:

```python
allow_origins=["http://localhost:ВАШЕ_ПОРТ"],
```

### Данные не загружаются
Проверьте, что:
1. API сервер запущен на порту 8000
2. В `data/raw/` есть файл отчёта
3. В консоли браузера нет ошибок CORS

## 🎉 Готово!

Современная дизайн-система полностью интегрирована с вашим проектом аналитики СВК. 

**Сравнение:**
- **Было:** Streamlit с базовым UI
- **Стало:** Modern Next.js дашборд с профессиональным дизайном

**Преимущества:**
- ✅ Быстрый responsive интерфейс
- ✅ Dark mode
- ✅ Продвинутые фильтры
- ✅ Экспорт данных
- ✅ Расширяемая архитектура
- ✅ TypeScript type safety
