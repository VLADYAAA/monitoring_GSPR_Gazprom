import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    pipeline
)
from datasets import Dataset
import torch
import json

# Загрузка данных из JSON
with open('reviews.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Преобразуем в DataFrame
df = pd.DataFrame(data)

# Предобработка данных
# Создаем целевую переменную для эмоциональной окраски (пример: 1-2 = негатив, 3 = нейтрал, 4-5 = позитив)
def map_sentiment(rating):
    if rating <= 2:
        return 0  # негатив
    elif rating == 3:
        return 1  # нейтрал
    else:
        return 2  # позитив

# Создаем маппинг тегов в числовые labels
unique_tags = df['reviewTag'].unique()
tag_to_id = {tag: idx for idx, tag in enumerate(unique_tags)}

df['sentiment_label'] = df['rating'].apply(map_sentiment)
df['topic_label'] = df['reviewTag'].map(tag_to_id)

print("Распределение эмоций:", df['sentiment_label'].value_counts())
print("Распределение тем:", df['topic_label'].value_counts())