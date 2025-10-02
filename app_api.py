from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
from transformers import pipeline

app = FastAPI()

# Загрузка моделей
sentiment_classifier = pipeline(
    "text-classification",
    model="./sentiment_model",
    tokenizer="./sentiment_model",
    device=0 if torch.cuda.is_available() else -1
)

topic_classifier = pipeline(
    "text-classification",
    model="./topic_model",
    tokenizer="./topic_model",
    device=0 if torch.cuda.is_available() else -1
)

class Review(BaseModel):
    id: int
    text: str

class Prediction(BaseModel):
    id: int
    topics: List[str]
    sentiments: List[str]

class RequestBody(BaseModel):
    data: List[Review]

@app.post("/predict", response_model=List[Prediction])
async def predict(request: RequestBody):
    predictions = []
    for review in request.data:
        sentiment_result = sentiment_classifier(review.text)[0]
        topic_result = topic_classifier(review.text)[0]

        predictions.append(
            Prediction(
                id=review.id,
                topics=[topic_result["label"]],
                sentiments=[sentiment_result["label"]]
            )
        )
    return predictions
