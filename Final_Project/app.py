
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Weather Intelligence | AI Prediction",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "rain_prediction_pipeline.pkl"

artifact = joblib.load(MODEL_PATH)
model = artifact["pipeline"]
threshold = artifact["threshold"]

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    h1, h2, h3 {
        font-weight: 700;
    }

    .hero {
        background: linear-gradient(120deg, #102A43, #176B87);
        padding: 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 34px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #D9EAF2;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-card {
        background: var(--secondary-background-color);
        padding: 18px;
        border-radius: 14px;
        margin-bottom: 12px;
        border: 1px solid rgba(128,128,128,0.15);
    }

    .result-card {
        padding: 25px;
        border-radius: 16px;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.2);
        text-align: center;
    }

    .result-number {
        font-size: 42px;
        font-weight: 800;
        margin: 8px 0;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
    }

    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🌦️ Weather AI")
    st.caption("Machine Learning Prediction System")
    st.divider()

    st.markdown("### About the Model")
    st.write("**Algorithm:** LightGBM")
    st.write("**Task:** Binary Classification")
    st.write("**Target:** RainTomorrow")
    st.write(f"**Decision Threshold:** {threshold:.2f}")

    st.divider()
    st.markdown("### How it works")
    st.write(
        "Enter today's weather conditions. "
        "The trained machine learning model estimates "
        "the probability of rain tomorrow."
    )

    st.divider()
    st.caption("Developed as a Machine Learning project.")

# ---------------- MAIN HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🌦️ Weather Intelligence</h1>
    <p>AI-powered weather prediction system</p>
    <p>Predict the possibility of rainfall tomorrow using today's weather data.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Weather Information")
st.write("Enter the weather conditions below to generate an AI prediction.")

wind_directions = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
]

# ---------------- INPUT FORM ----------------
with st.form("weather_form"):

    st.markdown("#### 📍 Location and Date")

    c1, c2, c3 = st.columns(3)

    with c1:
        location = st.text_input("Location", value="Canberra")

    with c2:
        month = st.number_input(
            "Month", min_value=1, max_value=12, value=1
        )

    with c3:
        day = st.number_input(
            "Day", min_value=1, max_value=31, value=26
        )

    st.divider()
    st.markdown("#### 🌡️ Temperature and Rainfall")

    c1, c2, c3 = st.columns(3)

    with c1:
        min_temp = st.number_input("Minimum Temperature (°C)", value=15.1)
        max_temp = st.number_input("Maximum Temperature (°C)", value=32.9)

    with c2:
        temp_9am = st.number_input("Temperature at 9am (°C)", value=19.0)
        temp_3pm = st.number_input("Temperature at 3pm (°C)", value=30.3)

    with c3:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0)
        evaporation = st.number_input("Evaporation (mm)", min_value=0.0, value=0.0)
        sunshine = st.number_input(
            "Sunshine (hours)", min_value=0.0, max_value=24.0, value=8.0
        )

    st.divider()
    st.markdown("#### 💨 Wind Conditions")

    c1, c2, c3 = st.columns(3)

    with c1:
        wind_gust_dir = st.selectbox("Wind Gust Direction", wind_directions)
        wind_gust_speed = st.number_input("Wind Gust Speed (km/h)", min_value=0.0, value=50.0)

    with c2:
        wind_dir_9am = st.selectbox("Wind Direction at 9am", wind_directions)
        wind_speed_9am = st.number_input("Wind Speed at 9am (km/h)", min_value=0.0, value=7.0)

    with c3:
        wind_dir_3pm = st.selectbox("Wind Direction at 3pm", wind_directions)
        wind_speed_3pm = st.number_input("Wind Speed at 3pm (km/h)", min_value=0.0, value=11.0)

    st.divider()
    st.markdown("#### 💧 Humidity, Pressure and Cloud")

    c1, c2, c3 = st.columns(3)

    with c1:
        humidity_9am = st.number_input(
            "Humidity at 9am (%)", min_value=0.0, max_value=100.0, value=73.0
        )
        humidity_3pm = st.number_input(
            "Humidity at 3pm (%)", min_value=0.0, max_value=100.0, value=32.0
        )

    with c2:
        pressure_9am = st.number_input("Pressure at 9am (hPa)", value=1015.5)
        pressure_3pm = st.number_input("Pressure at 3pm (hPa)", value=1011.8)

    with c3:
        cloud_9am = st.number_input(
            "Cloud at 9am (oktas)", min_value=0.0, max_value=8.0, value=7.0
        )
        cloud_3pm = st.number_input(
            "Cloud at 3pm (oktas)", min_value=0.0, max_value=8.0, value=4.0
        )

    st.divider()

    rain_today = st.selectbox("🌧️ Did it rain today?", ["No", "Yes"])

    submitted = st.form_submit_button(
        "🔍 Predict Rain Tomorrow",
        type="primary"
    )

# ---------------- PREDICTION ----------------
if submitted:

    input_data = pd.DataFrame([{
        "Location": location,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "Evaporation": evaporation,
        "Sunshine": sunshine,
        "WindGustDir": wind_gust_dir,
        "WindGustSpeed": wind_gust_speed,
        "WindDir9am": wind_dir_9am,
        "WindDir3pm": wind_dir_3pm,
        "WindSpeed9am": wind_speed_9am,
        "WindSpeed3pm": wind_speed_3pm,
        "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm,
        "Pressure9am": pressure_9am,
        "Pressure3pm": pressure_3pm,
        "Cloud9am": cloud_9am,
        "Cloud3pm": cloud_3pm,
        "Temp9am": temp_9am,
        "Temp3pm": temp_3pm,
        "RainToday": rain_today,
        "Month": int(month),
        "Day": int(day)
    }])

    probability = model.predict_proba(input_data)[0][1]
    prediction = int(probability >= threshold)

    st.divider()
    st.markdown("## 📊 Prediction Results")

    result_col, probability_col = st.columns(2)

    with result_col:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown("### 🌧️ Rain Expected")
            st.markdown(
                '<div class="result-number">YES</div>',
                unsafe_allow_html=True
            )
            st.warning("Rain is expected tomorrow based on the model prediction.")
        else:
            st.markdown("### ☀️ No Rain Expected")
            st.markdown(
                '<div class="result-number">NO</div>',
                unsafe_allow_html=True
            )
            st.success("No rain is expected tomorrow based on the model prediction.")

        st.markdown('</div>', unsafe_allow_html=True)

    with probability_col:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### 🌧️ Rain Probability")
        st.markdown(
            f'<div class="result-number">{probability:.1%}</div>',
            unsafe_allow_html=True
        )
        st.progress(float(probability))
        st.caption(f"Decision threshold: {threshold:.0%}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.info(
        "The probability represents the model's estimated likelihood "
        "of rain tomorrow. The final classification uses the selected "
        "decision threshold."
    )

    with st.expander("View submitted weather data"):
        st.dataframe(input_data, use_container_width=True)

else:
    st.markdown("---")
    st.info("👆 Enter the weather information and click Predict Rain Tomorrow to see the results.")
