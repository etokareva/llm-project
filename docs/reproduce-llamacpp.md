# Воспроизведение результатов с llama.cpp

Итоговый прогон выполнен на Windows с Docker Desktop и WSL. Модели обслуживал `llama-server` (сборка 12167, коммит `240d68fcf`) через Vulkan.

В каталоге с `llama-server.exe` создай `models.ini`. Укажи абсолютные пути к весам Qwen3 4B Instruct 2507 Q4_K_M и BGE-M3. Имена секций совпадают с именами моделей в `.env` и используются при обращении к сохранённому кешу.

```ini
version = 1

[*]
device = Vulkan0
n-gpu-layers = 99
ctx-size = 8192
parallel = 1

[qwen3-4b-instruct-2507-q4_K_M]
model = C:/models/qwen3-4b-instruct-2507-q4_K_M.gguf
jinja = true

[bge-m3]
model = C:/models/bge-m3.gguf
embedding = true
pooling = cls
```

Запусти сервер в PowerShell из каталога с `llama-server.exe` и оставь его работать на время проверки.

```powershell
.\llama-server.exe --models-preset .\models.ini --models-max 2 --host 127.0.0.1 --port 8080
```

В WSL открой корень клона. Если `.env` ещё нет, создай его командой `cp .env.example .env`. Установи в файле следующие значения.

```dotenv
LLM_BASE_URL=http://host.docker.internal:8080/v1
LLM_MODEL=qwen3-4b-instruct-2507-q4_K_M
EMBED_MODEL=bge-m3
ARTIFACTS_DIR=artifacts
```

Выполни в WSL из корня репозитория:

```bash
make up
make index
make test
make lint
make eval-all
make eval-extraction
```

Результаты сохраняются в `artifacts/`, кеш модели — в `artifacts/llm_cache/`. При проверке чистого клона все 109 обращений при индексации обслужены из кеша. Прошли 215 тестов и Ruff, итоговые метрики воспроизвелись.