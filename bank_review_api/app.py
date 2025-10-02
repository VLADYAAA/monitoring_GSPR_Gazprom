from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Optional
import logging
import time
import asyncio
from models.topic_classifier import TopicClassifier
from models.sentiment_analyzer import SentimentAnalyzer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Модели данных
class ReviewItem(BaseModel):
    id: int
    text: str

    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

class PredictionRequest(BaseModel):
    data: List[ReviewItem]

    @validator('data')
    def validate_data_length(cls, v):
        if len(v) > 250:
            raise ValueError('Maximum 250 reviews per request')
        return v

class PredictionResponseItem(BaseModel):
    id: int
    topics: List[str]
    sentiments: List[str]

class PredictionResponse(BaseModel):
    predictions: List[PredictionResponseItem]

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

# Инициализация приложения
app = FastAPI(
    title="Bank Review Analysis API",
    description="API для анализа тематики и тональности банковских отзывов",
    version="1.0.0"
)

# Глобальные переменные для моделей
topic_classifier = None
sentiment_analyzer = None

@app.on_event("startup")
async def startup_event():
    """Загрузка моделей при старте приложения"""
    global topic_classifier, sentiment_analyzer
    logger.info("Loading models...")
    
    try:
        topic_classifier = TopicClassifier()
        sentiment_analyzer = SentimentAnalyzer()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "Bank Review Analysis API", 
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "models_loaded": topic_classifier is not None and sentiment_analyzer is not None
    }

@app.post("/predict", 
          response_model=PredictionResponse,
          responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def predict(request: PredictionRequest):
    """
    Основной endpoint для предсказания тем и тональности отзывов
    
    - Поддерживает до 250 отзывов за запрос
    - Возвращает темы и тональности для каждого отзыва
    - Время обработки < 3 минут
    """
    start_time = time.time()
    
    try:
        # Проверка загружены ли модели
        if topic_classifier is None or sentiment_analyzer is None:
            raise HTTPException(status_code=503, detail="Models are not loaded yet")
        
        # Обработка отзывов
        predictions = []
        
        for review in request.data:
            try:
                # Определяем темы
                topics = await topic_classifier.predict(review.text)
                
                # Определяем тональность для каждой темы
                sentiments = []
                for topic in topics:
                    sentiment = await sentiment_analyzer.analyze_sentiment_for_topic(
                        review.text, topic
                    )
                    sentiments.append(sentiment)
                
                predictions.append(PredictionResponseItem(
                    id=review.id,
                    topics=topics,
                    sentiments=sentiments
                ))
                
            except Exception as e:
                logger.error(f"Error processing review {review.id}: {e}")
                # В случае ошибки для конкретного отзыва, возвращаем пустые списки
                predictions.append(PredictionResponseItem(
                    id=review.id,
                    topics=[],
                    sentiments=[]
                ))
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(predictions)} reviews in {processing_time:.2f}s")
        
        return PredictionResponse(predictions=predictions)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Обработчики ошибок
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Validation error", "details": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "details": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)