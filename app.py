import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import random

# Настройки
st.set_page_config(page_title="Клиентские настроения", layout="wide")
st.image("orig.png", width=120)
st.title("Дашборд клиентских настроений и проблем")

# Загрузка данных пользователем
uploaded_file = st.file_uploader("Загрузите JSON с отзывами", type="json")

if uploaded_file is not None:
    try:
        reviews_data = json.load(uploaded_file)
        
        # ДЕБАГ: покажем что загрузили
        st.sidebar.write("📁 Загружено отзывов:", len(reviews_data["data"]))
        st.sidebar.write("📝 Пример отзыва:", reviews_data["data"][0] if reviews_data["data"] else "Нет данных")

        # Создаем DataFrame из загруженных данных
        reviews_df = pd.DataFrame(reviews_data["data"])

        # Добавляем случайные даты для демонстрации временного интервала
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2025, 5, 31)
        date_range = (end_date - start_date).days

        reviews_df['date'] = [
            start_date + timedelta(days=random.randint(0, date_range))
            for _ in range(len(reviews_df))
        ]

        # Получение предсказаний через API
        def fetch_predictions(data):
            url = "http://localhost:8000/predict"
            try:
                response = requests.post(url, json=data, timeout=30)
                if response.status_code == 200:
                    return response.json()
                else:
                    st.error(f"Ошибка при запросе к API: {response.status_code} - {response.text}")
                    return None
            except Exception as e:
                st.error(f"Ошибка подключения к API: {e}")
                return None

        predictions_data = fetch_predictions(reviews_data)

        if predictions_data:
            # ДЕБАГ: покажем что вернуло API
            st.sidebar.write("🎯 Получено предсказаний:", len(predictions_data))
            st.sidebar.write("📊 Пример предсказания:", predictions_data[0] if predictions_data else "Нет данных")
            
            # Преобразование предсказаний
            flat_data = []
            for pred in predictions_data:
                # Проверяем что есть темы и тональности
                if pred.get("topics") and pred.get("sentiments"):
                    for topic, sentiment in zip(pred["topics"], pred["sentiments"]):
                        flat_data.append({
                            "id": pred["id"],
                            "topic": topic,
                            "sentiment": sentiment
                        })
                else:
                    st.warning(f"Нет тем или тональностей для отзыва ID: {pred.get('id')}")
            
            flat_df = pd.DataFrame(flat_data)
            
            # ДЕБАГ: покажем статистику
            if not flat_df.empty:
                st.sidebar.write("🎭 Уникальные тональности:", flat_df["sentiment"].unique().tolist())
                st.sidebar.write("🏷️ Уникальные темы:", flat_df["topic"].unique().tolist())
                st.sidebar.write("📈 Распределение тональностей:", flat_df["sentiment"].value_counts().to_dict())

            # Объединение с отзывами
            merged_df = reviews_df.merge(flat_df, on="id", how="left")
            
            # Проверяем что данные есть после объединения
            if merged_df.empty:
                st.error("❌ Нет данных после объединения с предсказаниями")
                st.stop()

            # Боковая панель с фильтрами
            st.sidebar.header("🔧 Фильтры")

            # Временной интервал
            min_date = merged_df["date"].min().date()
            max_date = merged_df["date"].max().date()

            date_range = st.sidebar.date_input(
                "Выберите временной интервал:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            if len(date_range) == 2:
                start_date_filter, end_date_filter = date_range
                filtered_df = merged_df[
                    (merged_df["date"].dt.date >= start_date_filter) &
                    (merged_df["date"].dt.date <= end_date_filter)
                ]
            else:
                filtered_df = merged_df

            # 1. Отображение списка продуктов/услуг
            st.header("📋 Список продуктов/услуг")
            
            # Получаем уникальные темы (исключаем NaN)
            topics = sorted([str(topic) for topic in filtered_df["topic"].unique() if pd.notna(topic)])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Всего тем", len(topics))
            with col2:
                st.metric("Всего отзывов", len(filtered_df))
            with col3:
                st.metric("Период", f"{date_range[0]} - {date_range[1]}" if len(date_range) == 2 else "Не выбран")

            st.write("**Обнаруженные темы:**", ", ".join(topics) if topics else "Нет тем")

            # 2. Детальная статистика по каждому продукту/услуге
            if topics:
                st.header("📊 Анализ по продуктам/услугам")

                for i, topic in enumerate(topics):
                    with st.expander(f"🔍 {topic}", expanded=False):
                        # Фильтруем данные по теме
                        topic_data = filtered_df[filtered_df["topic"] == topic]
                        total_reviews = len(topic_data)

                        if total_reviews > 0:
                            # Получаем распределение тональностей
                            sentiment_counts = topic_data["sentiment"].value_counts()

                            # Создаем данные для графиков
                            chart_data = []
                            # Используем реальные тональности из данных
                            unique_sentiments = sentiment_counts.index.tolist()
                            
                            for sentiment in unique_sentiments:
                                count = sentiment_counts.get(sentiment, 0)
                                percent = round((count / total_reviews) * 100, 1) if total_reviews > 0 else 0
                                chart_data.append({
                                    'sentiment': sentiment,
                                    'count': count,
                                    'percent': percent
                                })

                            chart_df = pd.DataFrame(chart_data)

                            col1, col2, col3 = st.columns(3)

                            # --- Pie chart ---
                            with col1:
                                st.subheader("Процентное распределение")

                                pie_df = chart_df[chart_df['count'] > 0]

                                if not pie_df.empty:
                                    # Автоматическая цветовая схема
                                    fig_pie = px.pie(
                                        pie_df,
                                        values='count',
                                        names='sentiment',
                                        hole=0.4,
                                        title=f"Распределение тональностей: {topic}"
                                    )
                                    fig_pie.update_traces(
                                        textinfo='percent+label',
                                        hovertemplate='<b>%{label}</b><br>Количество: %{value}<br>Доля: %{percent}'
                                    )
                                    fig_pie.update_layout(
                                        showlegend=True,
                                        legend=dict(orientation="v", yanchor="auto", y=1, xanchor="auto", x=1)
                                    )
                                    st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{topic}_{i}")
                                else:
                                    st.info("Нет данных для диаграммы")

                            # --- Bar chart ---
                            with col2:
                                st.subheader("Абсолютное распределение")

                                # Создаем барчарт с правильными данными
                                if not chart_df.empty:
                                    # Определяем цвета для разных тональностей
                                    color_map = {
                                        'positive': '#00CC66',
                                        'negative': '#FF3333', 
                                        'neutral': '#FFCC00'
                                    }
                                    
                                    # Создаем список цветов для каждого столбца
                                    colors = [color_map.get(sentiment, '#CCCCCC') for sentiment in chart_df['sentiment']]
                                    
                                    fig_bar = go.Figure()
                                    fig_bar.add_trace(go.Bar(
                                        x=chart_df['sentiment'],
                                        y=chart_df['count'],
                                        text=chart_df['count'],
                                        textposition='outside',
                                        marker_color=colors,
                                        hovertemplate='<b>%{x}</b><br>Количество: %{y}<extra></extra>'
                                    ))
                                    
                                    fig_bar.update_layout(
                                        title=f"Количество отзывов: {topic}",
                                        xaxis_title="Тональность",
                                        yaxis_title="Количество отзывов",
                                        plot_bgcolor='white',
                                        showlegend=False,
                                        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='lightgray')
                                    )
                                    st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{topic}_{i}")
                                else:
                                    st.info("Нет данных для столбчатой диаграммы")

                            # --- Цифры ---
                            with col3:
                                st.subheader("Статистика")
                                st.metric("Всего отзывов", total_reviews)
                                st.write("---")

                                # Иконки для разных тональностей
                                icon_map = {'positive': '🟢', 'neutral': '🟡', 'negative': '🔴'}
                                
                                for row in chart_data:
                                    icon = icon_map.get(row['sentiment'], '⚫')
                                    st.metric(
                                        label=f"{icon} {row['sentiment']}",
                                        value=f"{row['count']} отзывов",
                                        delta=f"{row['percent']}%"
                                    )

                            # --- Ключевые аспекты ---
                            st.subheader("📝 Примеры отзывов")

                            # Берем по 2 примера каждой тональности
                            for sentiment in unique_sentiments:
                                sentiment_data = topic_data[topic_data["sentiment"] == sentiment]
                                if not sentiment_data.empty:
                                    st.write(f"**{sentiment.upper()}:**")
                                    # Берем тексты отзывов
                                    sample_texts = sentiment_data["text"].head(2).tolist()
                                    for text in sample_texts:
                                        st.write(f"• {text}")
                                    st.write("")

                        else:
                            st.info("Нет отзывов по данной теме за выбранный период")
            else:
                st.warning("⚠️ Не обнаружено тем для анализа")

            # --- Общая статистика по тональностям ---
            st.header("📈 Общая статистика")
            
            if not filtered_df.empty:
                # Общее распределение тональностей
                overall_sentiment = filtered_df["sentiment"].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if not overall_sentiment.empty:
                        fig_overall = px.pie(
                            values=overall_sentiment.values,
                            names=overall_sentiment.index,
                            title="Общее распределение тональностей"
                        )
                        st.plotly_chart(fig_overall, use_container_width=True)
                    else:
                        st.info("Нет данных для общей круговой диаграммы")
                
                with col2:
                    # Топ тем (исключаем NaN)
                    topic_counts = filtered_df["topic"].value_counts()
                    if not topic_counts.empty:
                        # Убираем NaN значения
                        topic_counts = topic_counts[topic_counts.index.notna()]
                        top_topics = topic_counts.head(10)
                        
                        if not top_topics.empty:
                            fig_topics = px.bar(
                                x=top_topics.values,
                                y=top_topics.index,
                                orientation='h',
                                title="Топ-10 самых упоминаемых тем",
                                labels={'x': 'Количество отзывов', 'y': 'Темы'}
                            )
                            fig_topics.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig_topics, use_container_width=True)
                        else:
                            st.info("Нет данных для топ-тем")
                    else:
                        st.info("Нет данных по темам")

            else:
                st.warning("Нет данных для отображения общей статистики")

        else:
            st.error("Не удалось получить предсказания от API.")
            
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
        st.error(f"Тип ошибки: {type(e).__name__}")
else:
    st.info("Пожалуйста, загрузите файл с отзывами в формате JSON.")