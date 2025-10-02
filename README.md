# 🏦 Bank Review Analysis API

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Микросервис для автоматического анализа банковских отзывов. Определяет тематику и эмоциональную окраску текстов на русском языке.

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