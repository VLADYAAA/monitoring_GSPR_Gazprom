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
        
        reviews_df['date'] = [start_date + timedelta(days=random.randint(0, date_range)) 
                             for _ in range(len(reviews_df))]
        
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
            topics = sorted(filtered_df["topic"].unique())
            
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
            
            for topic in topics:
                with st.expander(f"🔍 {topic}", expanded=False):
                    topic_data = filtered_df[filtered_df["topic"] == topic]
                    total_reviews = len(topic_data)
                    
                    if total_reviews > 0:
                        # Процентное и абсолютное распределение
                        sentiment_counts = topic_data["sentiment"].value_counts()
                        sentiment_percent = (sentiment_counts / total_reviews * 100).round(1)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        # ПРАВИЛЬНАЯ КРУГОВАЯ ДИАГРАММА
                        with col1:
                            st.subheader("Процентное распределение")
                            
                            # Создаем данные для диаграммы
                            pie_data = []
                            for sentiment in ['positive', 'neutral', 'negative']:
                                count = sentiment_counts.get(sentiment, 0)
                                percent = sentiment_percent.get(sentiment, 0)
                                if count > 0:
                                    pie_data.append({
                                        'sentiment': sentiment,
                                        'count': count,
                                        'percent': percent
                                    })
                            
                            if pie_data:
                                pie_df = pd.DataFrame(pie_data)
                                
                                fig_pie = px.pie(
                                    pie_df,
                                    values='count',
                                    names='sentiment',
                                    color='sentiment',
                                    color_discrete_map={
                                        'positive': '#2E8B57',
                                        'negative': '#DC143C', 
                                        'neutral': '#FFD700'
                                    },
                                    hole=0.3
                                )
                                
                                fig_pie.update_traces(
                                    textinfo='percent+label',
                                    hovertemplate='<b>%{label}</b><br>Количество: %{value}<br>Процент: %{percent}'
                                )
                                
                                st.plotly_chart(fig_pie, use_container_width=True)
                            else:
                                st.info("Нет данных для отображения")
                        
                        # ПРАВИЛЬНАЯ СТОЛБЧАТАЯ ДИАГРАММА
                        with col2:
                            st.subheader("Абсолютное распределение")
                            
                            # Создаем данные для столбчатой диаграммы
                            bar_data = []
                            for sentiment in ['positive', 'neutral', 'negative']:
                                count = sentiment_counts.get(sentiment, 0)
                                bar_data.append({
                                    'sentiment': sentiment,
                                    'count': count
                                })
                            
                            bar_df = pd.DataFrame(bar_data)
                            
                            fig_bar = px.bar(
                                bar_df,
                                x='sentiment',
                                y='count',
                                color='sentiment',
                                color_discrete_map={
                                    'positive': '#2E8B57',
                                    'negative': '#DC143C', 
                                    'neutral': '#FFD700'
                                },
                                text='count'
                            )
                            
                            fig_bar.update_layout(
                                showlegend=False,
                                xaxis_title="Тональность",
                                yaxis_title="Количество отзывов",
                                yaxis=dict(range=[0, max(bar_df['count'].max() + 1, 5)])
                            )
                            
                            fig_bar.update_traces(
                                texttemplate='%{text}',
                                textposition='outside'
                            )
                            
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # СТАТИСТИКА
                        with col3:
                            st.subheader("Статистика")
                            st.metric("Всего отзывов", total_reviews)
                            
                            for sentiment in ['positive', 'neutral', 'negative']:
                                count = sentiment_counts.get(sentiment, 0)
                                percent = sentiment_percent.get(sentiment, 0)
                                
                                # Выбираем иконку в зависимости от тональности
                                icon = {
                                    'positive': '🟢',
                                    'neutral': '⚪', 
                                    'negative': '🔴'
                                }.get(sentiment, '⚪')
                                
                                st.metric(
                                    f"{icon} {sentiment.capitalize()}",
                                    f"{count} ({percent}%)"
                                )
                        
                        # КЛЮЧЕВЫЕ АСПЕКТЫ
                        st.subheader("📝 Ключевые аспекты")
                        
                        positive_aspects = topic_data[topic_data["sentiment"] == "positive"]["text"].head(3)
                        negative_aspects = topic_data[topic_data["sentiment"] == "negative"]["text"].head(3)
                        neutral_aspects = topic_data[topic_data["sentiment"] == "neutral"]["text"].head(3)
                        
                        col4, col5, col6 = st.columns(3)
                        
                        with col4:
                            if len(positive_aspects) > 0:
                                st.write("**✅ Что нравится:**")
                                for text in positive_aspects:
                                    st.write(f"• {text}")
                            else:
                                st.write("**✅ Что нравится:**")
                                st.write("Нет положительных отзывов")
                        
                        with col5:
                            if len(neutral_aspects) > 0:
                                st.write("**⚪ Нейтральные отзывы:**")
                                for text in neutral_aspects:
                                    st.write(f"• {text}")
                            else:
                                st.write("**⚪ Нейтральные отзывы:**")
                                st.write("Нет нейтральных отзывов")
                        
                        with col6:
                            if len(negative_aspects) > 0:
                                st.write("**❌ Что не нравится:**")
                                for text in negative_aspects:
                                    st.write(f"• {text}")
                            else:
                                st.write("**❌ Что не нравится:**")
                                st.write("Нет отрицательных отзывов")
                    
                    else:
                        st.info("Нет отзывов по данной теме за выбранный период")
            
            # 4. Динамика во времени
            st.header("📈 Динамика во времени")
            
            # Выбор тем для анализа
            selected_topics = st.multiselect(
                "Выберите темы для анализа динамики:",
                options=topics,
                default=topics[:2] if len(topics) >= 2 else topics
            )
            
            if selected_topics:
                # Подготовка данных для временных рядов
                time_data = filtered_df[filtered_df["topic"].isin(selected_topics)].copy()
                time_data['month'] = time_data['date'].dt.to_period('M').astype(str)
                
                # 4.1 Динамика тональностей по продуктам
                st.subheader("Динамика тональностей по продуктам")
                
                # Создаем subplot для каждой темы
                fig_tonality = make_subplots(
                    rows=len(selected_topics), 
                    cols=1,
                    subplot_titles=[f"Динамика тональностей: {topic}" for topic in selected_topics],
                    vertical_spacing=0.1
                )
                
                for i, topic in enumerate(selected_topics, 1):
                    topic_time_data = time_data[time_data["topic"] == topic]
                    
                    if not topic_time_data.empty:
                        # Группируем по месяцам и тональностям
                        monthly_sentiment = topic_time_data.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
                        
                        # Рассчитываем проценты
                        monthly_total = monthly_sentiment.sum(axis=1)
                        monthly_percent = monthly_sentiment.div(monthly_total, axis=0) * 100
                        
                        for sentiment in ['positive', 'neutral', 'negative']:
                            if sentiment in monthly_percent.columns:
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
                
                fig_tonality.update_layout(height=300 * len(selected_topics), title_text="Динамика долей тональностей")
                st.plotly_chart(fig_tonality, use_container_width=True)
                
                # 4.2 Динамика количества отзывов
                st.subheader("Динамика количества отзывов")
                
                fig_volume = go.Figure()
                
                for topic in selected_topics:
                    topic_time_data = time_data[time_data["topic"] == topic]
                    
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
                st.plotly_chart(fig_volume, use_container_width=True)
                
                # Сводная таблица динамики
                st.subheader("📋 Сводная таблица динамики")
                
                pivot_table = time_data.groupby(['month', 'topic', 'sentiment']).size().unstack(fill_value=0)
                st.dataframe(pivot_table)

        else:
            st.error("Не удалось получить предсказания от API.")
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
else:
    st.info("Пожалуйста, загрузите файл с отзывами в формате JSON.")