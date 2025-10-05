import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
from pathlib import Path

MODEL_DIR  = Path("model")
MODEL_PATH = MODEL_DIR / "rain_pkg.joblib"
FILE_ID    = st.secrets["MODEL_FILE_ID"]

def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    return pred

def download_model():
    # якщо моделі немає - завантажити
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        st.write("⬇️ Downloading model from Google Drive...")
        gdown.download(url, str(MODEL_PATH), quiet=False)

# завантажити вхідні данні
download_model
aussie_rain = joblib.load(MODEL_PATH)

#aussie_rain = joblib.load('model/rain_pkg.joblib')
model = aussie_rain['model']
imputer = aussie_rain['imputer']
scaler = aussie_rain['scaler']
encoder = aussie_rain['encoder']
numeric_cols = aussie_rain['numeric_cols']
encoded_cols = aussie_rain['encoded_cols']
categorical_cols = aussie_rain['categorical_cols']

raw_df = pd.read_csv("dataset/weatherAUS.csv")
locations_list = sorted(raw_df["Location"].dropna().unique())

# Заголовок застосунку
st.title('Передбачення дощу')
st.markdown('Модель для передбачення дощу в Австралії' )

with st.form("weather_form"):
    st.subheader("Введіть параметри погоди сьогодні:")
    date = st.date_input("Date")
    location = st.selectbox("Location", options=locations_list)

    col1, col2, col3 = st.columns(3)
    with col1:
        min_temp = st.number_input("MinTemp", value=14.2)
        max_temp = st.number_input("MaxTemp", value=20.2)
        temp_9am = st.number_input("Temp9am", value=15.7)
        temp_3pm = st.number_input("Temp3pm", value=24.0)
        pressure_9am = st.number_input("Pressure9am", value=1004.8)
        pressure_3pm = st.number_input("Pressure3pm", value=1001.5)

    with col2:
        wind_gust_dir = st.selectbox("WindGustDir", raw_df["WindGustDir"].dropna().unique())
        wind_gust_speed = st.number_input("WindGustSpeed", value=52.0)
        wind_dir_9am = st.selectbox("WindDir9am", raw_df["WindDir9am"].dropna().unique())
        wind_dir_3pm = st.selectbox("WindDir3pm", raw_df["WindDir3pm"].dropna().unique())
        wind_speed_9am = st.number_input("WindSpeed9am", value=13.0)
        wind_speed_3pm = st.number_input("WindSpeed3pm", value=20.0)

    with col3:
        humidity_9am = st.number_input("Humidity9am", value=89.0)
        humidity_3pm = st.number_input("Humidity3pm", value=75.0)
        cloud_9am = st.number_input("Cloud9am", value=8.0)
        cloud_3pm = st.number_input("Cloud3pm", value=9.0)
        rainfall = st.number_input("Rainfall", value=10.2)
        evaporation = st.number_input("Evaporation", value=4.2)
        sunshine = st.number_input("Sunshine (може бути NaN)", value=0.0)
        rain_today = st.selectbox("RainToday", ["Yes", "No"])

    submitted = st.form_submit_button("Прогноз")

# Кнопка для прогнозування
if submitted:
    new_input = {
        'Date': str(date),
        'Location': location,
        'MinTemp': min_temp,
        'MaxTemp': max_temp,
        'Rainfall': rainfall,
        'Evaporation': evaporation,
        'Sunshine': np.nan if sunshine == 0 else sunshine,
        'WindGustDir': wind_gust_dir,
        'WindGustSpeed': wind_gust_speed,
        'WindDir9am': wind_dir_9am,
        'WindDir3pm': wind_dir_3pm,
        'WindSpeed9am': wind_speed_9am,
        'WindSpeed3pm': wind_speed_3pm,
        'Humidity9am': humidity_9am,
        'Humidity3pm': humidity_3pm,
        'Pressure9am': pressure_9am,
        'Pressure3pm': pressure_3pm,
        'Cloud9am': cloud_9am,
        'Cloud3pm': cloud_3pm,
        'Temp9am': temp_9am,
        'Temp3pm': temp_3pm,
        'RainToday': rain_today
    }
    # Викликаємо функцію прогнозування
    result = predict_input(new_input)
    if result == 'Yes':
        st.error(f"Прогноз на завтра: **Дощ**")
    else:
        st.success(f"Прогноз на завтра: **Без дощу**")
 
