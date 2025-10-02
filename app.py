import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json

# Настройки
st.set_page_config(page_title="Клиентские настроения", layout="wide")
st.image("orig.png", width=120)
st.title("Дашборд клиентских настроений и проблем")

# Загрузка данных пользователем
uploaded_file = st.file_uploader("Загрузите JSON с отзывами", type="json")

if uploaded_file is not None:
    try:
        reviews_data = json.load(uploaded_file)
        
        # Создаем DataFrame из загруженных данных
        reviews_df = pd.DataFrame(reviews_data["data"])

        # Получение предсказаний через API
        def fetch_predictions(data):
            url = "http://localhost:8000/predict"
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Ошибка при запросе к API: {response.text}")
                return None

        predictions_data = fetch_predictions(reviews_data)

        if predictions_data:
            # Преобразование предсказаний
            flat_data = []
            for pred in predictions_data:
                for topic, sentiment in zip(pred["topics"], pred["sentiments"]):
                    flat_data.append({"id": pred["id"], "topic": topic, "sentiment": sentiment})
            flat_df = pd.DataFrame(flat_data)

            # Объединение с отзывами
            merged_df = reviews_df.merge(flat_df, on="id", how="left")

            # Отображение списка продуктов/услуг
            st.subheader("📋 Список продуктов/услуг")
            topics = merged_df["topic"].unique()
            st.write(list(topics))

            # Статистика по тональностям
            st.subheader("⚖️ Распределение тональностей по темам")
            sentiment_counts = merged_df.groupby(["topic", "sentiment"]).size().reset_index(name="count")
            fig = px.bar(sentiment_counts, x="topic", y="count", color="sentiment", barmode="stack")
            st.plotly_chart(fig, use_container_width=True)

            # Общая статистика по тональностям
            st.subheader("📊 Общее распределение тональностей")
            overall_sentiment = merged_df["sentiment"].value_counts()
            fig_pie = px.pie(values=overall_sentiment.values, names=overall_sentiment.index)
            st.plotly_chart(fig_pie, use_container_width=True)

            # Детализированный список отзывов
            st.subheader("📝 Детализированные отзывы")
            for topic in topics:
                with st.expander(f"Тема: {topic}"):
                    topic_reviews = merged_df[merged_df["topic"] == topic]
                    for _, row in topic_reviews.iterrows():
                        sentiment_color = {
                            "positive": "🟢",
                            "negative": "🔴", 
                            "neutral": "⚪"
                        }.get(row["sentiment"], "⚪")
                        
                        st.write(f"{sentiment_color} **ID {row['id']}**: {row['text']}")

        else:
            st.error("Не удалось получить предсказания от API.")
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
else:
    st.info("Пожалуйста, загрузите файл с отзывами в формате JSON.")