import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
from datetime import datetime

# Настройки
st.set_page_config(page_title="Клиентские настроения", layout="wide")
st.image("orig.png", width=120)
st.title("Дашборд клиентских настроений и проблем")

# Загрузка данных пользователем
uploaded_file = st.file_uploader("Загрузите JSON с отзывами", type="json")

if uploaded_file is not None:
    try:
        reviews_data = json.load(uploaded_file)

        # Получение предсказаний через API
        def fetch_predictions(data):
            url = "http://localhost:8000/predict"
            response = requests.post(url, json={"data": data})
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

            # Фильтр по датам
            date_range = st.sidebar.slider(
                "Выберите временной интервал:",
                min_value=merged_df["date"].min().to_pydatetime(),
                max_value=merged_df["date"].max().to_pydatetime(),
                value=(merged_df["date"].min().to_pydatetime(), merged_df["date"].max().to_pydatetime())
            )
            filtered_df = merged_df[(merged_df["date"] >= pd.to_datetime(date_range[0])) &
                                    (merged_df["date"] <= pd.to_datetime(date_range[1]))]

            # Отображение списка продуктов/услуг
            st.subheader("📋 Список продуктов/услуг")
            topics = filtered_df["topic"].unique()
            st.write(topics)

            # Статистика по тональностям
            st.subheader("⚖️ Распределение тональностей по темам")
            sentiment_counts = filtered_df.groupby(["topic", "sentiment"]).size().reset_index(name="count")
            fig = px.bar(sentiment_counts, x="topic", y="count", color="sentiment", barmode="stack")
            st.plotly_chart(fig, use_container_width=True)

            # Динамика во времени
            st.subheader("📈 Динамика тональностей во времени")
            time_series = filtered_df.groupby([filtered_df["date"].dt.date, "sentiment"]).size().reset_index(name="count")
            fig2 = px.line(time_series, x="date", y="count", color="sentiment", markers=True)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("Не удалось получить предсказания от API.")
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
else:
    st.info("Пожалуйста, загрузите файл с отзывами в формате JSON.")
