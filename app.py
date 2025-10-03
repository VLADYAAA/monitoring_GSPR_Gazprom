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
                    flat_data.append({
                        "id": pred["id"],
                        "topic": topic,
                        "sentiment": sentiment
                    })
            flat_df = pd.DataFrame(flat_data)

            # Объединение с отзывами
            merged_df = reviews_df.merge(flat_df, on="id", how="left")

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
            
            # Убедимся, что topics - это строки и отсортируем их как строки
            topics = sorted([str(topic) for topic in filtered_df["topic"].unique()])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Всего тем", len(topics))
            with col2:
                st.metric("Всего отзывов", len(filtered_df))
            with col3:
                st.metric("Период", f"{date_range[0]} - {date_range[1]}")

            st.write("**Обнаруженные темы:**", ", ".join(topics))

            # 2. Детальная статистика по каждому продукту/услуге
            st.header("📊 Анализ по продуктам/услугам")

            for i, topic in enumerate(topics):
                with st.expander(f"🔍 {topic}", expanded=False):
                    # Преобразуем topic обратно в строку для фильтрации
                    topic_str = str(topic)
                    topic_data = filtered_df[filtered_df["topic"].astype(str) == topic_str]
                    total_reviews = len(topic_data)

                    if total_reviews > 0:
                        # ИСПРАВЛЕНИЕ: Правильно рассчитываем распределение тональностей для КОНКРЕТНОЙ ТЕМЫ
                        sentiment_counts = topic_data["sentiment"].value_counts()

                        chart_data = []
                        for sentiment in ['positive', 'neutral', 'negative']:
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

                            # ИСПРАВЛЕНИЕ: Используем chart_df который содержит данные по конкретной теме
                            pie_df = chart_df[chart_df['count'] > 0]

                            if not pie_df.empty:
                                fig_pie = px.pie(
                                    pie_df,
                                    values='count',
                                    names='sentiment',
                                    color='sentiment',
                                    color_discrete_map={
                                        'positive': '#00CC66',
                                        'negative': '#FF3333',
                                        'neutral': '#FFCC00'
                                    },
                                    hole=0.4
                                )
                                fig_pie.update_traces(
                                    textinfo='label+percent',
                                    hovertemplate='<b>%{label}</b><br>Количество: %{value}<br>Доля: %{percent}'
                                )
                                # Добавляем уникальный ключ
                                st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{topic}_{i}")
                            else:
                                st.info("Нет данных для диаграммы")

                        # --- Bar chart ---
                        with col2:
                            st.subheader("Абсолютное распределение")

                            # ИСПРАВЛЕНИЕ: Используем chart_df который содержит данные по конкретной теме
                            fig_bar = px.bar(
                                chart_df,
                                x='sentiment',
                                y='count',
                                color='sentiment',
                                color_discrete_map={
                                    'positive': '#00CC66',
                                    'negative': '#FF3333',
                                    'neutral': '#FFCC00'
                                },
                                text='count'
                            )
                            fig_bar.update_traces(
                                texttemplate='%{text}',
                                textposition='outside'
                            )
                            fig_bar.update_layout(
                                xaxis_title="Тональность",
                                yaxis_title="Количество",
                                plot_bgcolor='white',
                                showlegend=False
                            )
                            # Добавляем уникальный ключ
                            st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{topic}_{i}")

                        # --- Цифры ---
                        with col3:
                            st.subheader("Статистика")
                            st.metric("Всего отзывов", total_reviews)
                            st.write("---")

                            icon_map = {'positive': '🟢', 'neutral': '⚪', 'negative': '🔴'}

                            for row in chart_data:
                                st.metric(
                                    label=f"{icon_map[row['sentiment']]} {row['sentiment'].capitalize()}",
                                    value=f"{row['count']} отзывов",
                                    delta=f"{row['percent']}%"
                                )

                        # --- Ключевые аспекты ---
                        st.subheader("📝 Ключевые аспекты")

                        positive_aspects = topic_data[topic_data["sentiment"] == "positive"]["text"].head(3)
                        neutral_aspects = topic_data[topic_data["sentiment"] == "neutral"]["text"].head(3)
                        negative_aspects = topic_data[topic_data["sentiment"] == "negative"]["text"].head(3)

                        col4, col5, col6 = st.columns(3)

                        with col4:
                            st.write("**✅ Что нравится:**")
                            if len(positive_aspects) > 0:
                                for text in positive_aspects:
                                    st.write(f"• {text}")
                            else:
                                st.write("Нет положительных отзывов")

                        with col5:
                            st.write("**⚪ Нейтральные отзывы:**")
                            if len(neutral_aspects) > 0:
                                for text in neutral_aspects:
                                    st.write(f"• {text}")
                            else:
                                st.write("Нет нейтральных отзывов")

                        with col6:
                            st.write("**❌ Что не нравится:**")
                            if len(negative_aspects) > 0:
                                for text in negative_aspects:
                                    st.write(f"• {text}")
                            else:
                                st.write("Нет отрицательных отзывов")

                    else:
                        st.info("Нет отзывов по данной теме за выбранный период")

            # --- Динамика во времени ---
            st.header("📈 Динамика во времени")

            selected_topics = st.multiselect(
                "Выберите темы для анализа динамики:",
                options=topics,
                default=topics[:2] if len(topics) >= 2 else topics,
                key="topic_selector"
            )

            if selected_topics:
                # Преобразуем selected_topics в строки для фильтрации
                selected_topics_str = [str(topic) for topic in selected_topics]
                time_data = filtered_df[filtered_df["topic"].astype(str).isin(selected_topics_str)].copy()
                
                # Создаем месяц в правильном формате для сортировки
                time_data['month'] = time_data['date'].dt.strftime('%Y-%m')
                time_data = time_data.sort_values('date')

                # --- Динамика тональностей ---
                st.subheader("Динамика тональностей по продуктам")
                
                if len(selected_topics) > 0:
                    fig_tonality = make_subplots(
                        rows=len(selected_topics),
                        cols=1,
                        subplot_titles=[f"{topic}" for topic in selected_topics],
                        vertical_spacing=0.1
                    )

                    for i, topic in enumerate(selected_topics, 1):
                        topic_str = str(topic)
                        topic_time_data = time_data[time_data["topic"].astype(str) == topic_str]

                        if not topic_time_data.empty:
                            # Группируем по месяцам без сортировки (уже отсортированы по дате)
                            monthly_sentiment = (
                                topic_time_data
                                .groupby(['month', 'sentiment'])
                                .size()
                                .unstack(fill_value=0)
                            )
                            
                            # Заполняем отсутствующие колонки
                            for sentiment in ['positive', 'neutral', 'negative']:
                                if sentiment not in monthly_sentiment.columns:
                                    monthly_sentiment[sentiment] = 0

                            monthly_total = monthly_sentiment.sum(axis=1)
                            monthly_percent = (monthly_sentiment.div(monthly_total, axis=0) * 100).fillna(0)

                            for sentiment in ['positive', 'neutral', 'negative']:
                                fig_tonality.add_trace(
                                    go.Scatter(
                                        x=monthly_percent.index,
                                        y=monthly_percent[sentiment],
                                        name=f"{topic} - {sentiment}",
                                        mode='lines+markers',
                                        legendgroup=topic,
                                        showlegend=(i == 1)
                                    ),
                                    row=i, col=1
                                )

                    fig_tonality.update_layout(height=300 * len(selected_topics),
                                               title_text="Динамика долей тональностей")
                    # Добавляем уникальный ключ
                    st.plotly_chart(fig_tonality, use_container_width=True, key="tonality_dynamic")

                    # --- Динамика количества ---
                    st.subheader("Динамика количества отзывов")

                    fig_volume = go.Figure()

                    for topic in selected_topics:
                        topic_str = str(topic)
                        topic_time_data = time_data[time_data["topic"].astype(str) == topic_str]

                        if not topic_time_data.empty:
                            monthly_counts = topic_time_data.groupby('month').size()

                            fig_volume.add_trace(
                                go.Scatter(
                                    x=monthly_counts.index,
                                    y=monthly_counts.values,
                                    name=topic,
                                    mode='lines+markers'
                                )
                            )

                    fig_volume.update_layout(
                        xaxis_title="Месяц",
                        yaxis_title="Количество отзывов",
                        title="Динамика абсолютного числа упоминаний"
                    )
                    # Добавляем уникальный ключ
                    st.plotly_chart(fig_volume, use_container_width=True, key="volume_dynamic")

                    # --- Сводная таблица ---
                    st.subheader("📋 Сводная таблица динамики")
                    pivot_table = (
                        time_data
                        .groupby(['month', 'topic', 'sentiment'])
                        .size()
                        .unstack(fill_value=0)
                    )
                    # Заполняем отсутствующие колонки
                    for sentiment in ['positive', 'neutral', 'negative']:
                        if sentiment not in pivot_table.columns:
                            pivot_table[sentiment] = 0
                    
                    st.dataframe(pivot_table)

        else:
            st.error("Не удалось получить предсказания от API.")
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
else:
    st.info("Пожалуйста, загрузите файл с отзывами в формате JSON.")