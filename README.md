# News Topic Classifier — MLOps Курсовой проект

Сервис классификации тематики новостей (AG News: World / Sports / Business / Sci/Tech)  
Реализован полный жизненный цикл ML-модели с использованием ClearML.

---

## 🔗 Публичные ссылки

| Сервис | URL | Описание |
|--------|-----|----------|
| **ClearML UI** | http://81.26.187.144:8080 | Эксперименты, датасеты, модели |
| **ClearML API** | http://81.26.187.144:8008 | REST API сервера |
| **ClearML Files** | http://81.26.187.144:8081 | Хранилище артефактов |
| **Inference Endpoint** | http://81.26.187.144:9000/serve/news-topic | HTTP POST, тело: `{"text": "..."}` |
| **Health Check** | http://81.26.187.144:9000/health | Статус serving-сервера |
| **Streamlit UI** | http://81.26.187.144:8501 | Пользовательский интерфейс |

---

## ✅ Выполненные этапы

### Этап 0 — Инфраструктура

- [x] Развёрнут ClearML Server (API :8008, Web :8080, Files :8081)
- [x] Настроен SDK через `clearml-init` (`.env` с ключами)
- [x] Поднят ClearML Agent, подключён к очереди `students`
- [x] Agent виден в ClearML UI → Workers
- [x] Task отправляется в очередь и выполняется агентом (не локально)

**Проверка агента:**
```bash
clearml-agent list
```

---

### Этап 1 — Dataset

- [x] Подготовлен датасет AG News (4 класса, 5000 записей: 4000 train / 1000 test)
- [x] Создан ClearML Dataset через SDK (`dataset/upload_dataset.py`)
- [x] Файлы загружены: `train.csv`, `test.csv`
- [x] Версия зафиксирована: `1.0`
- [x] Dataset виден в ClearML UI → Datasets
- [x] Обучение использует `dataset_id` из ClearML

**Запуск загрузки датасета:**
```bash
python dataset/upload_dataset.py
```

**Dataset ID сохраняется в:** `dataset/dataset_id.txt`

**Структура датасета:**

| Поле | Тип | Описание |
|------|-----|----------|
| `text` | string | Текст новости |
| `label` | int | Числовая метка (0–3) |
| `label_name` | string | World / Sports / Business / Sci/Tech |

---

### Этап 2 — Обучение через Agent + логирование

- [x] Скрипт `train/train.py` создаёт ClearML Task
- [x] Логируются гиперпараметры (max_features, ngram_max, C, solver, ...)
- [x] Логируются метрики: accuracy, f1_macro, f1_weighted, f1 по каждому классу
- [x] Логируется Confusion Matrix как изображение
- [x] Модель сохраняется как artifact (vectorizer, classifier, model_bundle)
- [x] Датасет загружается из ClearML по dataset_id
- [x] Запуск через очередь `students`, выполнение агентом
- [x] Проведено минимум 2 эксперимента с разными параметрами

**Запуск экспериментов:**
```bash
python scripts/run_experiments.py
```

**Эксперименты:**

| Параметр | Эксперимент 1: `logreg_baseline` | Эксперимент 2: `logreg_bigram_highC` |
|----------|----------------------------------|--------------------------------------|
| max_features | 20 000 | 50 000 |
| ngram_max | 1 (unigram) | 2 (bigram) |
| C | 1.0 | 5.0 |
| max_iter | 300 | 500 |
| solver | lbfgs | lbfgs |

**В ClearML UI видно:**
- 2 эксперимента в проекте `news_topic_classification`
- Различия в параметрах (вкладка Configuration)
- Различия в метриках (вкладка Scalars)
- Confusion Matrix (вкладка Plots)
- Артефакты: vectorizer.pkl, classifier.pkl, model_bundle.zip

---

### Этап 3 — Model Registry

- [x] Выбрана лучшая модель по метрике f1_macro
- [x] Модель зарегистрирована в ClearML Model Registry
- [x] Модель имеет версию
- [x] Проставлены теги: `production`, `logreg`, `ag_news`
- [x] Сохранены метрики в описании модели (accuracy, f1_macro)

**Регистрация лучшей модели:**
```bash
python scripts/register_best_model.py
```

Скрипт автоматически:
1. Находит все завершённые задачи обучения
2. Выбирает задачу с максимальным f1_macro
3. Публикует модель в Registry
4. Сохраняет `best_model_id.txt` и `best_task_id.txt`

**В ClearML UI:** Models → Registry → `news_topic_logreg`

---

### Этап 4 — Inference Endpoint

- [x] Поднят Flask serving-сервер (`serving/serve.py`)
- [x] Модель загружается из ClearML Registry по `MODEL_ID` (не локальный .pkl)
- [x] HTTP endpoint доступен: `POST /serve/news-topic`
- [x] Health check: `GET /health`

**Запуск вручную:**
```bash
MODEL_ID=<id> TASK_ID=<id> python serving/serve.py
```

**Запуск через Docker:**
```bash
docker compose -f docker-compose.prod.yml up -d serving
```

**Примеры запросов:**

```bash
# World — международные новости
curl -X POST http://81.26.187.144:9000/serve/news-topic \
  -H "Content-Type: application/json" \
  -d '{"text": "World leaders gather in Geneva for emergency climate summit."}'

# Sports — спорт
curl -X POST http://81.26.187.144:9000/serve/news-topic \
  -H "Content-Type: application/json" \
  -d '{"text": "Real Madrid defeats Barcelona in a thrilling El Clasico match."}'

# Business — бизнес
curl -X POST http://81.26.187.144:9000/serve/news-topic \
  -H "Content-Type: application/json" \
  -d '{"text": "The stock market crashed after the Federal Reserve raised interest rates."}'

# Sci/Tech — наука и технологии
curl -X POST http://81.26.187.144:9000/serve/news-topic \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple unveils new AI-powered chip for the iPhone 17."}'

# Health check
curl http://81.26.187.144:9000/health
```

**Формат ответа:**
```json
{
  "prediction": 1,
  "label": "Sports",
  "confidence": 0.9823,
  "probabilities": {
    "World":    0.0041,
    "Sports":   0.9823,
    "Business": 0.0098,
    "Sci/Tech": 0.0038
  }
}
```

**Схема загрузки модели:**
```
MODEL_ID (env)
    ↓
Model.get_local_copy()  ← ClearML Registry
    ↓
vectorizer.pkl + classifier.pkl  ← из zip-бандла или артефактов Task
    ↓
/serve/news-topic готов принимать запросы
```

---

### Этап 5 — UI (Streamlit)

- [x] Поле ввода текста новости
- [x] Кнопка Predict
- [x] Отображается label (тематика) с emoji
- [x] Отображается confidence (уверенность модели, %)
- [x] Отображается latency (время ответа в мс)
- [x] Обрабатываются ошибки: недоступный endpoint, таймаут, HTTP ошибки
- [x] UI работает **только через HTTP** к serving endpoint, модель не загружается напрямую
- [x] Быстрые примеры для каждого класса
- [x] Настраиваемый URL endpoint в sidebar
- [x] Проверка статуса endpoint из UI

**Запуск вручную:**
```bash
SERVING_URL=http://81.26.187.144:9000/serve/news-topic \
  streamlit run ui/app.py --server.port 8501
```

**Запуск через Docker:**
```bash
docker compose -f docker-compose.prod.yml up -d ui
```

**Доступен по адресу:** http://81.26.187.144:8501

---

## 🏗️ Архитектура проекта

```
┌─────────────────┐     HTTP POST      ┌──────────────────────┐
│  Streamlit UI   │ ─────────────────► │   Flask Serving      │
│  :8501          │  /serve/news-topic │   :9000              │
└─────────────────┘                    └──────────┬───────────┘
                                                  │ загрузка модели
                                                  ▼
                                       ┌──────────────────────┐
                                       │   ClearML Registry   │
                                       │   :8008 / :8081      │
                                       └──────────────────────┘
```

```
news-topic-classification/
├── app_config/
│   ├── __init__.py          # Загрузка конфига + env override
│   └── config.yaml          # Все параметры проекта
├── dataset/
│   └── upload_dataset.py    # Загрузка AG News в ClearML Dataset
├── train/
│   └── train.py             # Обучение + логирование в ClearML
├── serving/
│   ├── serve.py             # Flask inference server
│   └── preprocess.py        # ClearML Serving preprocess module
├── scripts/
│   ├── run_experiments.py   # Запуск всех экспериментов
│   └── register_best_model.py # Регистрация лучшей модели
├── ui/
│   └── app.py               # Streamlit UI
├── .github/
│   └── workflows/
│       └── deploy.yml       # CD: автодеплой при push в main
├── docker-compose.prod.yml  # Продакшн docker-compose
├── Dockerfile.serving       # Docker образ для serving
├── Dockerfile.ui            # Docker образ для UI
├── requirements.txt         # Общие зависимости
├── requirements.serving.txt # Зависимости serving (+ flask, gunicorn)
├── requirements.ui.txt      # Зависимости UI (+ streamlit)
└── .env                     
```

---

## 🚀 Быстрый старт (полный цикл)

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd news-topic-classification

# 2. Создать .env (скопировать шаблон и заполнить)
cp .env.example .env

# 3. Загрузить датасет в ClearML
python dataset/upload_dataset.py

# 4. Запустить эксперименты через Agent
python scripts/run_experiments.py

# 5. Зарегистрировать лучшую модель
python scripts/register_best_model.py

# 6. Запустить serving + UI через Docker
docker compose -f docker-compose.prod.yml up -d --build

# 7. Открыть UI
open http://81.26.187.144:8501
```

---

## ⚙️ Конфигурация

Все параметры в `app_config/config.yaml`. Env переменные имеют приоритет.

| Переменная | Описание | Пример |
|------------|----------|--------|
| `MODEL_ID` | ID модели из ClearML Registry | `0ce5d561...` |
| `TASK_ID` | ID задачи с артефактами | `5a41c0bc...` |
| `SERVING_URL` | URL endpoint для UI | `http://81.26.187.144:9000/serve/news-topic` |
| `CLEARML_API_HOST` | Адрес ClearML API | `http://81.26.187.144:8008` |

---

## 📊 Итоги

| Этап | Статус | Баллы |
|------|--------|-------|
| Инфраструктура (ClearML Server + Agent) | ✅ Выполнено | 1 |
| Dataset (AG News в ClearML Datasets) | ✅ Выполнено | 2 |
| Training + Agent (2 эксперимента, метрики, артефакты) | ✅ Выполнено | 3 |
| Model Registry (публикация, теги, версия) | ✅ Выполнено | 2 |
| Inference (Flask endpoint, загрузка из Registry) | ✅ Выполнено | 3 |
| UI (Streamlit, HTTP only, latency, обработка ошибок) | ✅ Выполнено | 1 |
| **Итого** | | **12 / 12** |
```