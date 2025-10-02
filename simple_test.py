from transformers import pipeline
import torch

def quick_test():
    """Быстрый тест обученных моделей"""
    print("БЫСТРЫЙ ТЕСТ ОБУЧЕННЫХ МОДЕЛЕЙ")
    print("=" * 40)
    
    # Загружаем модели
    sentiment_analyzer = pipeline(
        "text-classification",
        model="./sentiment_model",
        tokenizer="./sentiment_model"
    )
    
    topic_analyzer = pipeline(
        "text-classification", 
        model="./topic_model",
        tokenizer="./topic_model"
    )
    
    # Тестовые тексты
    test_texts = [
        "Это просто ужасный сервис, никогда больше не обращусь!",
        "Нормально, но есть небольшие проблемы с приложением", 
        "Отлично! Быстро решили все вопросы, рекомендую этот банк!",
        "Карта удобная, но банкоматов маловато в нашем районе"
    ]
    
    print("Результаты анализа:")
    print("-" * 40)
    
    for i, text in enumerate(test_texts, 1):
        sentiment = sentiment_analyzer(text)[0]
        topic = topic_analyzer(text)[0]
        
        print(f"{i}. {text}")
        print(f"   🎭 Эмоция: {sentiment['label']} ({sentiment['score']:.3f})")
        print(f"   📁 Тема: {topic['label']} ({topic['score']:.3f})")
        print()

if __name__ == "__main__":
    quick_test()