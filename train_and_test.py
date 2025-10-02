import pandas as pd
import numpy as np
import json
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    pipeline
)
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

class ReviewAnalyzerTrainer:
    def __init__(self, model_name="cointegrated/rubert-tiny2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sentiment_model = None
        self.topic_model = None
        self.sentiment_trainer = None
        self.topic_trainer = None
        
    def load_and_prepare_data(self, json_file_path):
        """Загрузка и подготовка данных из JSON"""
        print("Загрузка данных...")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.df = pd.DataFrame(data)
        print(f"Загружено {len(self.df)} отзывов")
        
        # Преобразуем рейтинг в эмоциональную окраску
        def map_sentiment(rating):
            if rating <= 2:
                return 0  # негатив
            elif rating == 3:
                return 1  # нейтрал
            else:
                return 2  # позитив
        
        # Создаем маппинг тегов
        self.unique_tags = self.df['reviewTag'].unique()
        self.tag_to_id = {tag: idx for idx, tag in enumerate(self.unique_tags)}
        self.id_to_tag = {idx: tag for tag, idx in self.tag_to_id.items()}
        
        self.df['sentiment_label'] = self.df['rating'].apply(map_sentiment)
        self.df['topic_label'] = self.df['reviewTag'].map(self.tag_to_id)
        
        print("Распределение эмоций:")
        print(self.df['sentiment_label'].value_counts())
        print("\nРаспределение тем:")
        print(self.df['reviewTag'].value_counts())
        
        return self.df
    
    def tokenize_data(self, texts):
        """Токенизация текстов"""
        return self.tokenizer(
            texts, 
            padding="max_length", 
            truncation=True, 
            max_length=256,
            return_tensors="pt"
        )
    
    def train_sentiment_model(self):
        """Обучение модели для определения эмоций"""
        print("\n" + "="*50)
        print("ОБУЧЕНИЕ МОДЕЛИ ЭМОЦИОНАЛЬНОЙ ОКРАСКИ")
        print("="*50)
        
        # Создаем datasets
        sentiment_dataset = Dataset.from_pandas(
            self.df[['text', 'sentiment_label']].rename(columns={'sentiment_label': 'labels'})
        )
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
        
        tokenized_sentiment = sentiment_dataset.map(tokenize_function, batched=True)
        
        # Делим на train/test
        train_test = tokenized_sentiment.train_test_split(test_size=0.3, seed=42)
        train_dataset = train_test['train']
        eval_dataset = train_test['test']
        
        # Создаем модель
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            id2label={0: "negative", 1: "neutral", 2: "positive"},
            label2id={"negative": 0, "neutral": 1, "positive": 2}
        )
        
        # Параметры обучения
        training_args = TrainingArguments(
            output_dir="./sentiment_model",
            num_train_epochs=4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./sentiment_logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=2,
        )
        
        # Создаем тренер
        self.sentiment_trainer = Trainer(
            model=self.sentiment_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Обучаем
        print("Начинаем обучение...")
        self.sentiment_trainer.train()
        
        # Сохраняем модель
        self.sentiment_trainer.save_model("./sentiment_model")
        self.tokenizer.save_pretrained("./sentiment_model")
        print("Модель эмоций сохранена в './sentiment_model'")
        
        return self.sentiment_trainer
    
    def train_topic_model(self):
        """Обучение модели для определения темы"""
        print("\n" + "="*50)
        print("ОБУЧЕНИЕ МОДЕЛИ ОПРЕДЕЛЕНИЯ ТЕМЫ")
        print("="*50)
        
        # Создаем datasets
        topic_dataset = Dataset.from_pandas(
            self.df[['text', 'topic_label']].rename(columns={'topic_label': 'labels'})
        )
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
        
        tokenized_topic = topic_dataset.map(tokenize_function, batched=True)
        
        # Делим на train/test
        train_test = tokenized_topic.train_test_split(test_size=0.3, seed=42)
        train_dataset = train_test['train']
        eval_dataset = train_test['test']
        
        # Создаем модель
        self.topic_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.unique_tags),
            id2label=self.id_to_tag,
            label2id=self.tag_to_id
        )
        
        # Параметры обучения
        training_args = TrainingArguments(
            output_dir="./topic_model",
            num_train_epochs=4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./topic_logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=2,
        )
        
        # Создаем тренер
        self.topic_trainer = Trainer(
            model=self.topic_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Обучаем
        print("Начинаем обучение...")
        self.topic_trainer.train()
        
        # Сохраняем модель
        self.topic_trainer.save_model("./topic_model")
        self.tokenizer.save_pretrained("./topic_model")
        print("Модель тем сохранена в './topic_model'")
        
        return self.topic_trainer

class ReviewAnalyzer:
    def __init__(self, sentiment_model_path, topic_model_path):
        print("Загрузка моделей для анализа...")
        self.sentiment_classifier = pipeline(
            "text-classification",
            model=sentiment_model_path,
            tokenizer=sentiment_model_path,
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.topic_classifier = pipeline(
            "text-classification", 
            model=topic_model_path,
            tokenizer=topic_model_path,
            device=0 if torch.cuda.is_available() else -1
        )
        print("Модели загружены!")
    
    def analyze_review(self, text):
        """Анализ одного отзыва"""
        sentiment_result = self.sentiment_classifier(text)[0]
        topic_result = self.topic_classifier(text)[0]
        
        return {
            'text': text,
            'sentiment': {
                'label': sentiment_result['label'],
                'score': round(sentiment_result['score'], 4)
            },
            'topic': {
                'label': topic_result['label'],
                'score': round(topic_result['score'], 4)
            }
        }
    
    def analyze_batch(self, texts):
        """Анализ нескольких отзывов"""
        results = []
        for text in texts:
            results.append(self.analyze_review(text))
        return results
    
    def evaluate_models(self, test_texts, true_sentiments, true_topics):
        """Оценка качества моделей на тестовых данных"""
        print("\n" + "="*50)
        print("ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ")
        print("="*50)
        
        predictions = self.analyze_batch(test_texts)
        
        # Преобразуем предсказания в числовые метки для оценки
        sentiment_label_to_id = {"negative": 0, "neutral": 1, "positive": 2}
        
        pred_sentiments = [sentiment_label_to_id[pred['sentiment']['label']] for pred in predictions]
        pred_topics = [self.tag_to_id[pred['topic']['label']] for pred in predictions]
        
        # Оценка модели эмоций
        print("\n--- МОДЕЛЬ ЭМОЦИОНАЛЬНОЙ ОКРАСКИ ---")
        sentiment_accuracy = accuracy_score(true_sentiments, pred_sentiments)
        print(f"Accuracy: {sentiment_accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(true_sentiments, pred_sentiments, 
                                 target_names=["negative", "neutral", "positive"]))
        
        # Оценка модели тем
        print("\n--- МОДЕЛЬ ОПРЕДЕЛЕНИЯ ТЕМЫ ---")
        topic_accuracy = accuracy_score(true_topics, pred_topics)
        print(f"Accuracy: {topic_accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(true_topics, pred_topics, 
                                 target_names=[self.id_to_tag[i] for i in sorted(self.id_to_tag.keys())]))

def main():
    """Основная функция"""
    print("ЗАПУСК ОБУЧЕНИЯ И ТЕСТИРОВАНИЯ МОДЕЛЕЙ")
    print("="*60)
    
    # 1. Инициализация и подготовка данных
    trainer = ReviewAnalyzerTrainer()
    df = trainer.load_and_prepare_data(os.path.join(os.path.dirname(__file__), 'sravni_reviews_progress_22000.json'))
    
    # 2. Обучение моделей
    sentiment_trainer = trainer.train_sentiment_model()
    topic_trainer = trainer.train_topic_model()
    
    # 3. Тестирование моделей
    print("\n" + "="*50)
    print("ТЕСТИРОВАНИЕ МОДЕЛЕЙ")
    print("="*50)
    
    analyzer = ReviewAnalyzer("./sentiment_model", "./topic_model")
    
    # Тестовые примеры
    test_reviews = [
        "Отличный банк, все быстро и понятно! Обслуживание на высшем уровне.",
        "Ужасное обслуживание, долго решали простую проблему, сотрудники некомпетентны",
        "Обычный банк, ничего особенного. В целом нормально, но есть куда расти.",
        "В нашем городе--всего 3 банкомата( 3 места), где можно пополнить карту, два из которых периодически ломаются. Съездила в один конец--банкомат не работает, уехала --в другой--банкомат деньги не зачислил..потом ..поехала в сам банк..написала заявленме...ждали по графику инкассацию...внутренние расследования и перегонки...ответ зама..мол в течении 45 дней..подключилась к решению только мой добросовестный сотрудник( с кем работаю по вкладам)....обещали после инкассации--17-18.12.20....во вторник--22.12.20.....в среду 23.12.20 утром--точно !!! А сегодня---в понедельник 28.12.20, а может и через несколько дней....Ребята, вы, о чем?! У нас..что цифровизация только у Мишустина? И стоило мне снимать с пенсион.Мира (сбер)....?! и закидывать на Газпром.хэшбек карту, чтобы проездить ( весь хэшбе)--на бензин...и в итоге остаться без закупок к НОВ.ГОДУ. Спасибо Газпромбанку. Народ!!! Не совершайте моих ошибок, учтите мой печальный опыт.",
        "Мобильное приложение просто супер! Все функции работают стабильно, интерфейс интуитивный."
    ]
    
    print("Результаты анализа тестовых отзывов:")
    print("-" * 50)
    
    for i, review in enumerate(test_reviews, 1):
        result = analyzer.analyze_review(review)
        print(f"\n{i}. {review[:80]}...")
        print(f"   Эмоция: {result['sentiment']['label']} (уверенность: {result['sentiment']['score']:.3f})")
        print(f"   Тема: {result['topic']['label']} (уверенность: {result['topic']['score']:.3f})")
    
    # 4. Оценка на реальных данных (если есть разметка)
    print("\n" + "="*50)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ НА РЕАЛЬНЫХ ДАННЫХ")
    print("="*50)
    
    # Берем первые 3 примера из наших данных для демонстрации
    demo_texts = df['text'].head(3).tolist()
    demo_true_sentiments = df['sentiment_label'].head(3).tolist()
    demo_true_topics = df['topic_label'].head(3).tolist()
    
    print("Сравнение предсказаний с реальными метками:")
    for i, (text, true_sent, true_topic) in enumerate(zip(demo_texts, demo_true_sentiments, demo_true_topics)):
        result = analyzer.analyze_review(text)
        true_sentiment_label = ["negative", "neutral", "positive"][true_sent]
        true_topic_label = trainer.id_to_tag[true_topic]
        
        print(f"\n{i+1}. Реальный отзыв:")
        print(f"   Текст: {text[:100]}...")
        print(f"   РЕАЛЬНО: эмоция={true_sentiment_label}, тема={true_topic_label}")
        print(f"   ПРЕДСКАЗАНО: эмоция={result['sentiment']['label']}, тема={result['topic']['label']}")
        print(f"   СОВПАДЕНИЕ эмоций: {true_sentiment_label == result['sentiment']['label']}")
        print(f"   СОВПАДЕНИЕ тем: {true_topic_label == result['topic']['label']}")

if __name__ == "__main__":
    main()