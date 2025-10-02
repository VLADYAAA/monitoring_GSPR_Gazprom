
# 🏦 Bank Review Analysis API

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Микросервис для автоматического анализа банковских отзывов. Определяет тематику и эмоциональную окраску текстов на русском языке.
Ссылка на прототип http://46.173.16.84:8501/
Презентация приложена в репо.

## ✨ Возможности

- 🎯 **Мультитематическая классификация** - определение одной или нескольких тем в отзыве
- 😊 **Сентимент-анализ** - оценка эмоциональной окраски (положительно/нейтрально/отрицательно)
- ⚡ **Высокая производительность** - обработка до 250 отзывов за запрос
- 🔗 **REST API** - простой JSON интерфейс
- 🚀 **Готовность к продакшену** - обработка ошибок, валидация данных

## 🛠 Технологии

- **FastAPI** - современный, быстрый веб-фреймворк
- **Pydantic** - валидация данных и сериализация
- **Uvicorn** - ASGI сервер для production
- **Transformers** - ML модели для NLP (опционально)

## 📦 Быстрый старт

### Локальная разработка

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/yourusername/bank-review-api.git
cd bank-review-api

# 2. Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# или
venv\Scripts\activate     # Windows

# 3. Установите зависимости
pip install -r requirements.txt

# 4. Запустите сервер
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Сервис будет доступен по адресу: **http://localhost:8000**

### Бесплатный деплой на Render.com

1. **Fork этого репозитория** на GitHub
2. **Зарегистрируйтесь** на [Render.com](https://render.com)
3. **Создайте новый Web Service**
4. **Подключите ваш GitHub репозиторий**
5. **Настройки деплоя:**
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

Ваше API будет доступно по адресу: `https://your-api-name.onrender.com`

## 📚 Документация API

После запуска сервиса автоматически генерируется интерактивная документация:

**EndPoint for tests**: http://46.173.16.84:8000/


#### `POST /predict` - Анализ отзывов
**Request:**
```json
{
  "data": [
    {
      "id": 1,
      "text": "Отличное обслуживание в отделении, но мобильное приложение постоянно зависает."
    },
    {
      "id": 2,
      "text": "Кредитную карту оформили быстро, лимит хороший. Сотрудник вежливый."
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "id": 1,
      "topics": ["Обслуживание", "Мобильное приложение"],
      "sentiments": ["положительно", "отрицательно"]
    },
    {
      "id": 2,
      "topics": ["Кредитные карты", "Обслуживание"],
      "sentiments": ["положительно", "положительно"]
    }
  ]
}
```

## 🎯 Поддерживаемые темы

| Тема | Ключевые слова |
|------|----------------|
| **Обслуживание** | обслуживание, сотрудник, менеджер, консультация |
| **Мобильное приложение** | приложение, мобильный, онлайн, интернет-банк |
| **Кредитные карты** | кредитная карта, кредитка, карта, кредит |
| **Дебетовые карты** | дебетовая, карта, счет, перевод |
| **Кредиты** | кредит, заем, одобрили, процент |
| **Ипотека** | ипотека, ипотечный, недвижимость, квартира |
| **Вклады** | вклад, сбережения, накопления, процент |
| **Отделения** | отделение, офис, филиал, банкомат |

## 🚀 Примеры использования

### Python
```python
import requests
import json

API_URL = "https://your-api.onrender.com/predict"

def analyze_reviews(reviews):
    data = {
        "data": [
            {"id": i, "text": review}
            for i, review in enumerate(reviews)
        ]
    }
    
    response = requests.post(API_URL, json=data)
    return response.json()

# Пример использования
reviews = [
    "Очень понравилось обслуживание, но приложение глючит",
    "Кредит одобрили быстро, спасибо!"
]

results = analyze_reviews(reviews)
print(json.dumps(results, indent=2, ensure_ascii=False))
```

### JavaScript
```javascript
async function analyzeReviews(reviews) {
    const data = {
        data: reviews.map((text, index) => ({
            id: index + 1,
            text: text
        }))
    };
    
    const response = await fetch('https://your-api.onrender.com/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    });
    
    return await response.json();
}

// Пример использования
const reviews = [
    "Отличный сервис, быстро решили вопрос",
    "Ужасное обслуживание, никогда больше не обращусь"
];

analyzeReviews(reviews).then(console.log);
```

### cURL
```bash
curl -X POST "https://your-api.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "data": [
         {
           "id": 1,
           "text": "Прекрасное обслуживание, все быстро и четко"
         }
       ]
     }'
```


## 🐛 Troubleshooting

### Частые проблемы и решения

1. **Ошибка 422 - Validation Error**
   - Проверьте формат JSON
   - Убедитесь, что поле `text` не пустое

2. **Ошибка 503 - Models not loaded**
   - Подождите несколько секунд после старта сервиса
   - Проверьте лог-файлы

3. **Медленная обработка**
   - Уменьшите количество отзывов в запросе
   - Используйте более простую модель

4. **Ошибка деплоя на Render.com**
   - Убедитесь, что все файлы есть в репозитории
   - Проверьте корректность Procfile
   - Посмотрите логи билда в панели Render

### Ограничения

- ✅ Максимум **250 отзывов** за запрос
- ✅ Время обработки **< 3 минут**
- ✅ Поддержка **UTF-8** кодировки
- ✅ Автоматическая **обработка ошибок**

## 🔧 Разработка

### Структура проекта
```
bank-review-api/
├── app.py                 # Основное приложение FastAPI
├── requirements.txt      # Зависимости Python
├── runtime.txt          # Версия Python
├── Procfile            # Конфигурация для деплоя
└── README.md           # Документация
```


