import torch
from transformers import pipeline
from typing import List
import logging

logger = logging.getLogger(__name__)

class TopicClassifier:
    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="vicgalle/xlm-roberta-large-xnli-anli",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Список возможных тем (можно расширить)
        self.possible_topics = [
            "Обслуживание", "Дебетовые карты", "Удаленное обслуживание", 
            "Кредиты", "Кредитные карты", "Вклады и сбережения", "Ипотека",
            "Обмен валюты", "Автокредиты", "Мобильное приложение",
            "Бизнес услуги", "Денежные переводы", "Рефинансирование",
            "Технические проблемы", "Очереди", "Сотрудники", "Отделения"
        ]
    
    async def predict(self, text: str) -> List[str]:
        """Определяет темы текста"""
        try:
            result = self.classifier(
                text,
                self.possible_topics,
                multi_label=True,
                hypothesis_template="Данный текст относится к теме {}."
            )
            
            # Фильтруем темы с уверенностью > 0.2
            topics = []
            for label, score in zip(result['labels'], result['scores']):
                if score > 0.2:
                    topics.append(label)
            
            return topics if topics else ["Другое"]
            
        except Exception as e:
            logger.error(f"Topic classification error: {e}")
            return ["Другое"]

class SentimentAnalyzer:
    def __init__(self):
        self.classifier = pipeline(
            "sentiment-analysis",
            model="blanchefort/rubert-base-cased-sentiment",
            device=0 if torch.cuda.is_available() else -1
        )
        self.sentiment_mapping = {
            "positive": "положительно",
            "neutral": "нейтрально", 
            "negative": "отрицательно"
        }
    
    async def analyze_sentiment_for_topic(self, text: str, topic: str) -> str:
        """Определяет тональность для конкретной темы"""
        try:
            # В реальной реализации здесь была бы более сложная логика
            # для определения тональности по конкретной теме
            result = self.classifier(text[:512])[0]  # Ограничиваем длину
            
            english_sentiment = result['label']
            return self.sentiment_mapping.get(english_sentiment, "нейтрально")
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return "нейтрально"