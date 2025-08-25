# app.py
# ——————————————————————————————————————————————————————————————
# Mood Tracker AI – Akıllı Öneri + RandomForest + DistilBERT + Grafik
# Çalıştır: streamlit run app.py
# ——————————————————————————————————————————————————————————————
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from datetime import date
import json, os

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from transformers import pipeline
import plotly.express as px

# Page config
st.set_page_config(page_title="Mood Tracker AI", page_icon="🧠")
st.title("🧠 Mood Tracker AI – Akıllı Öneri Sistemi")

# ------------------------------------------------------------
# Örnek veri (CSV)
# ------------------------------------------------------------
CSV_TEXT = """date,mood_score,text,steps,sleep_hours,temp,weather
2025-08-01,6,"Sakin bir gündü",5000,7,25,sunny
2025-08-02,5,"Biraz yorgunum",4000,6,24,cloudy
2025-08-03,7,"Enerjik ve motiveyim",9000,8,30,sunny
2025-08-04,3,"Moralim bozuk ve stresliyim",2000,5,22,rainy
2025-08-05,8,"Harika bir gün geçirdim",11000,8,31,sunny
2025-08-06,6,"Odaklanabildim, fena değil",6500,7,27,cloudy
2025-08-07,4,"Biraz kaygılı hissediyorum",3000,6,23,cloudy
2025-08-08,7,"Üretken bir gündü",8000,7,29,sunny
2025-08-09,5,"Normal geçti",4500,6,26,cloudy
2025-08-10,3,"Uykusuz ve halsizim",1500,4,21,rainy
"""

DATA_FILE = "mood_log.json"

# ------------------------------------------------------------
# DistilBERT sentiment modeli (CPU)
# ------------------------------------------------------------
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

# ------------------------------------------------------------
# Öneri fonksiyonları
# ------------------------------------------------------------
def suggest_activity_bert(mood: float, sentiment_label: str) -> str:
    if mood < 4 or sentiment_label == "NEGATIVE":
        return "😔 Kısa bir mola ver; meditasyon veya nefes egzersizi iyi gelir."
    elif mood < 6.5:
        return "🙂 Hafif yürüyüş ve sevdiğin müzik iyi gelir."
    elif mood < 8.5:
        return "😃 Bugün küçük bir hedef belirle ve ilerle!"
    else:
        return "🔥 Süper enerji! Zor bir göreve odaklanıp ivme yakala."

def smart_suggestion(today_mood, sentiment_label, df_history):
    if df_history.empty:
        return suggest_activity_bert(today_mood, sentiment_label)
    
    recent_df = df_history.sort_values('date', key=pd.to_datetime).tail(7)
    recent_avg = recent_df['mood_score'].mean() if not recent_df.empty else None
    trend = recent_df['mood_score'].diff().mean() if len(recent_df) >= 2 else 0

    suggestion = suggest_activity_bert(today_mood, sentiment_label)
    if recent_avg is not None:
        if today_mood < recent_avg and trend < 0:
            suggestion += " ⚠️ Son günlerde ruh hâlin düştü, kendine ekstra mola ver."
        elif today_mood > recent_avg and trend > 0:
            suggestion += " ✅ Son günlerde ruh hâlin yükselişte, motivasyonunu koru!"
    return suggestion

# ------------------------------------------------------------
# RandomForest Model Eğitimi
# ------------------------------------------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv(StringIO(CSV_TEXT))
    X = df[["text", "steps", "sleep_hours", "temp", "weather"]]
    y = df["mood_score"].astype(float)

    numeric_features = ["steps", "sleep_hours", "temp"]
    categorical_features = ["weather"]
    text_feature = "text"

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("txt", TfidfVectorizer(max_features=300), text_feature),
        ],
        remainder="drop"
    )

    model_rf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("reg", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model_rf.fit(X_train, y_train)

    test_pred = model_rf.predict(X_test)
    mae = mean_absolute_error(y_test, test_pred) if len(X_test) else None
    return model_rf, mae

model_rf, mae = train_model()
st.success(f"Model hazır! Test MAE: {mae:.2f}" if mae is not None else "Model hazır!")

st.divider()

# ------------------------------------------------------------
# Kullanıcı girişi
# ------------------------------------------------------------
st.subheader("📖 Günlük Giriş")
with st.form("mood_form"):
    col1, col2 = st.columns(2)
    with col1:
        text = st.text_area("Günün özeti (kısa metin)", height=120, placeholder="Bugün nasıldı?")
        steps = st.number_input("Adım sayısı", min_value=0, max_value=50000, value=6000, step=500)
        sleep_hours = st.number_input("Uyku (saat)", min_value=0.0, max_value=14.0, value=7.0, step=0.5)
    with col2:
        temp = st.number_input("Sıcaklık (°C)", min_value=-30, max_value=50, value=25, step=1)
        weather = st.selectbox("Hava durumu", ["sunny", "cloudy", "rainy"])
        entry_date = st.date_input("Tarih", value=date.today())

    predict_btn = st.form_submit_button("Tahmin et ve öneri al")

# ------------------------------------------------------------
# Tahmin + DistilBERT Analizi + Akıllı Öneri
# ------------------------------------------------------------
if predict_btn:
    # RandomForest tahmini
    X_input = pd.DataFrame([{
        "text": text or "",
        "steps": steps,
        "sleep_hours": sleep_hours,
        "temp": temp,
        "weather": weather
    }])
    pred = float(model_rf.predict(X_input)[0])
    st.metric("AI Tahmini Ruh Hali (1–10)", f"{pred:.1f}")

    # DistilBERT analizi
    if text.strip():
        sentiment_result = sentiment_model(text)[0]
        top_sentiment = sentiment_result
        st.write(f"🤖 DistilBERT Analizi: **{top_sentiment['label']}** (%{top_sentiment['score']:.2f})")
    else:
        top_sentiment = {"label": "NEUTRAL", "score": 1.0}

    # JSON geçmiş verisini güvenli şekilde oku ve tarihleri normalize et
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        df_history = pd.DataFrame(data)

        # Tarihleri datetime yap, hatalı satırları düşür
        df_history['date'] = pd.to_datetime(df_history['date'], errors="coerce")
        df_history = df_history.dropna(subset=['date'])
    else:
        df_history = pd.DataFrame(columns=['date','mood_score','sentiment_label','sentiment_score'])

    # Akıllı öneri
    st.info(smart_suggestion(pred, top_sentiment['label'], df_history))

    # JSON kaydı
    mood_entry = {
        "date": entry_date.strftime("%Y-%m-%d"),
        "text": text,
        "mood_score": pred,
        "sentiment_label": top_sentiment['label'],
        "sentiment_score": float(top_sentiment['score'])
    }
    df_history = pd.concat([df_history, pd.DataFrame([mood_entry])], ignore_index=True)
    df_history.to_json(DATA_FILE, orient="records", indent=4, date_format='iso')

    # Görselleştirme – Son 7 gün
    df_vis = df_history.sort_values('date', key=pd.to_datetime).tail(7)

    if not df_vis.empty:
        fig = px.line(df_vis, x='date', y='mood_score', markers=True, title="Son 7 Gün Mood Skoru")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(df_vis, x='date', y='sentiment_score', color='sentiment_label',
                      title="Son 7 Gün NLP Sentiment Skoru", text='sentiment_label')
        st.plotly_chart(fig2, use_container_width=True)

st.caption("Not: Bu sürüm CPU uyumlu DistilBERT + RandomForest + Akıllı Öneri + Görselleştirme içerir.")
