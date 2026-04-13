# RAG System — Graph (Neo4j) vs Vector (ChromaDB)

Два режима поиска:
- **Graph RAG** — spaCy NER → Neo4j → обход графа
- **Vector RAG** — OpenRouter embeddings → ChromaDB → cosine similarity

FastAPI бэкенд + Gradio фронтенд.

---

## Установка

### 1. uv

```bash
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Зависимости

```bash
uv sync
```

---

## Запуск

### 1. Neo4j через Docker

```bash
docker compose up -d
```

Поднимает Neo4j на `bolt://localhost:7687` (логин `neo4j` / пароль `password`).

Веб-интерфейс Neo4j: http://localhost:7474

### 2. `.env`

```bash
cp .env_example .env
```

Заполнить `OPENAI_API_KEY` (ключ OpenRouter).

### 3. Backend + Frontend

**Терминал 1:**
```bash
uv run backend
```
→ http://localhost:8002

**Терминал 2:**
```bash
uv run frontend
```
→ http://localhost:7860

---

## Использование

1. Открыть http://localhost:7860
2. ⚙️ → выбрать режим: **graph** (Neo4j) или **vector** (ChromaDB)
3. Задать вопрос
4. Ответ + источники + метаданные отображаются в чате

---

## Параметры

| Параметр | Описание | Диапазон |
|----------|----------|----------|
| Top-K | Кол-во документов | 1–20 |
| Temperature | Креативность LLM | 0.0–1.0 |
| LLM Model | Модель OpenRouter | `openai/gpt-4o-mini` |
| Retrieval Mode | Режим поиска | `graph` / `vector` |
| Entity Context | Контекст сущностей (только graph) | on/off |

---

## API

Swagger UI: http://localhost:8000/docs

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| POST | `/query` | Запрос к RAG |
| GET | `/health` | Проверка здоровья |
| GET | `/stats` | Статистика системы |
| GET | `/graph/entities` | Топ сущностей из Neo4j |
| GET | `/graph/relations` | Топ связей из Neo4j |

---

## Остановка

```bash
docker compose down
```

## Troubleshooting

- **Cannot connect to backend** — запустите `uv run backend` в отдельном терминале
- **OPENAI_API_KEY not found** — проверьте `.env` файл
- **Neo4j connection failed** — убедитесь что `docker compose up -d` выполнен, проверьте http://localhost:7474
- **Embedding failed** — проверьте что ключ OpenRouter имеет доступ к embedding моделям
