import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

            # 1. РАСПРЕДЕЛЕНИЕ ТОНАЛЬНОСТЕЙ ПО ТЕМАМ (правильный вариант)
            st.subheader("⚖️ Распределение тональностей по темам")
            
            # Создаем сводную таблицу для правильного отображения
            pivot_data = merged_df.groupby(['topic', 'sentiment']).size().unstack(fill_value=0)
            
            # Создаем stacked bar chart
            fig_topic_sentiment = go.Figure()
            
            colors = {'positive': '#2E8B57', 'negative': '#DC143C', 'neutral': '#FFD700'}
            
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment in pivot_data.columns:
                    fig_topic_sentiment.add_trace(go.Bar(
                        name=sentiment.capitalize(),
                        x=pivot_data.index,
                        y=pivot_data[sentiment],
                        marker_color=colors[sentiment]
                    ))
            
            fig_topic_sentiment.update_layout(
                barmode='stack',
                xaxis_title="Темы",
                yaxis_title="Количество отзывов",
                legend_title="Тональность"
            )
            st.plotly_chart(fig_topic_sentiment, use_container_width=True)

            # 2. ОБЩЕЕ РАСПРЕДЕЛЕНИЕ ТОНАЛЬНОСТЕЙ
            st.subheader("📊 Общее распределение тональностей")
            
            sentiment_counts = merged_df['sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values, 
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map=colors
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

            # 3. РАСПРЕДЕЛЕНИЕ ТЕМ (без учета тональности)
            st.subheader("📈 Распределение тем")
            
            topic_counts = merged_df['topic'].value_counts()
            fig_topics = px.bar(
                x=topic_counts.index,
                y=topic_counts.values,
                labels={'x': 'Темы', 'y': 'Количество упоминаний'},
                color=topic_counts.values,
                color_continuous_scale='Viridis'
            )
            fig_topics.update_layout(showlegend=False)
            st.plotly_chart(fig_topics, use_container_width=True)

            # 4. ДЕТАЛИЗИРОВАННАЯ СТАТИСТИКА ПО ТЕМАМ
            st.subheader("🔍 Детализированная статистика по темам")
            
            # Создаем таблицу с детальной статистикой
            detailed_stats = merged_df.groupby('topic')['sentiment'].value_counts().unstack(fill_value=0)
            detailed_stats['Всего'] = detailed_stats.sum(axis=1)
            detailed_stats['% Положительных'] = (detailed_stats.get('positive', 0) / detailed_stats['Всего'] * 100).round(1)
            detailed_stats['% Отрицательных'] = (detailed_stats.get('negative', 0) / detailed_stats['Всего'] * 100).round(1)
            
            st.dataframe(detailed_stats.style.background_gradient(cmap='Blues'))

            # 5. ДЕТАЛИЗИРОВАННЫЙ СПИСОК ОТЗЫВОВ
            st.subheader("📝 Детализированные отзывы")
            
            selected_topic = st.selectbox("Выберите тему для просмотра отзывов:", topics)
            
            if selected_topic:
                topic_reviews = merged_df[merged_df["topic"] == selected_topic]
                
                # Фильтр по тональности
                selected_sentiment = st.selectbox("Фильтр по тональности:", 
                                                ["Все", "positive", "negative", "neutral"])
                
                if selected_sentiment != "Все":
                    topic_reviews = topic_reviews[topic_reviews["sentiment"] == selected_sentiment]
                
                st.write(f"**Отзывы по теме: {selected_topic}**")
                
                for _, row in topic_reviews.iterrows():
                    sentiment_color = {
                        "positive": "🟢",
                        "negative": "🔴", 
                        "neutral": "⚪"
                    }.get(row["sentiment"], "⚪")
                    
                    with st.container():
                        st.write(f"{sentiment_color} **ID {row['id']}** ({row['sentiment']}):")
                        st.write(f"_{row['text']}_")
                        st.divider()

        else:
            st.error("Не удалось получить предсказания от API.")
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
else:
    st.info("Пожалуйста, загрузите файл с отзывами в формате JSON.")