# ============================================================
#                    PLANTPULSE AI
#       REAL-TIME PLANT & ENVIRONMENTAL MONITORING
#       Arduino + PySerial + Random Forest + Live Weather
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import serial
import time
import joblib

from weather_api import get_live_weather
from ai_assistant.chatbot import ask_gemini, analyze_plant_image
from ai_assistant.context import build_plant_context


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PlantPulse AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SETTINGS
# ============================================================

ARDUINO_PORT = "COM12"
BAUD_RATE = 9600

MODEL_PATH = "random_forest_model.pkl"
FEATURE_PATH = "model_features.pkl"

# PlantPulse training dataset used to derive plant-specific ranges.
# Keep this CSV in the same folder as app.py.
DATASET_PATH = "PlantPulse_India_Crop_Stress_Dataset_75000.csv"


# ============================================================
# CUSTOM DARK THEME + ANIMATIONS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #0b1220;
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background-color: #070d18;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 17px;
    color: #94a3b8;
    margin-top: -8px;
}

.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #263244;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
    min-height: 130px;
}

.metric-title {
    color: #94a3b8;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #f8fafc;
    font-size: 30px;
    font-weight: 750;
    margin-top: 8px;
}

.metric-sub {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 5px;
}

.stress-box {
    background-color: #111827;
    border-radius: 18px;
    padding: 25px;
    border: 1px solid #263244;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
}

.stress-score {
    color: #f8fafc;
    font-size: 55px;
    font-weight: 800;
}

.stress-label {
    font-size: 18px;
    font-weight: 700;
}

.section-title {
    color: #f8fafc;
    font-size: 23px;
    font-weight: 750;
    margin-top: 10px;
    margin-bottom: 10px;
}

.recommendation {
    background-color: #111827;
    color: #e5e7eb;
    padding: 22px;
    border-radius: 16px;
    border-left: 5px solid #22c55e;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
}

h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
}

p {
    color: #cbd5e1;
}

button[data-baseweb="tab"] {
    color: #94a3b8;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #22c55e;
}

hr {
    border-color: #263244;
}


/* ============================================================
   ENVIRONMENT ANIMATION
   ============================================================ */

.environment-scene {
    position: relative;
    overflow: hidden;
    min-height: 260px;
    margin: 10px 0 20px 0;
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #263244;
    background: linear-gradient(
        180deg,
        #172033 0%,
        #0f172a 100%
    );
    box-shadow: 0px 4px 20px rgba(0,0,0,0.25);
    text-align: center;
}

.environment-scene.sunny {
    background: linear-gradient(
        180deg,
        #263b5c 0%,
        #172033 100%
    );
}

.environment-scene.rainy {
    background: linear-gradient(
        180deg,
        #1d293b 0%,
        #0f172a 100%
    );
}

.environment-scene.cloudy {
    background: linear-gradient(
        180deg,
        #263244 0%,
        #172033 100%
    );
}

.sun {
    position: absolute;
    top: 22px;
    right: 55px;
    font-size: 58px;
    animation: sunPulse 2s ease-in-out infinite;
    filter: drop-shadow(
        0 0 18px rgba(250,204,21,0.8)
    );
}

.cloud {
    position: absolute;
    top: 30px;
    left: 45px;
    font-size: 52px;
    animation: cloudMove 7s ease-in-out infinite;
}

.rain {
    position: absolute;
    inset: 0;
    pointer-events: none;
    font-size: 25px;
    letter-spacing: 18px;
    line-height: 1.7;
    opacity: 0.8;
    animation: rainFall 1s linear infinite;
}

.plant {
    position: relative;
    z-index: 2;
    display: inline-block;
    margin-top: 40px;
    font-size: 95px;
    transform-origin: bottom center;
    animation: plantSway 3s ease-in-out infinite;
}

.plant.thirsty,
.plant.heat,
.plant.stressed {
    animation: plantDroop 2.2s ease-in-out infinite;
}

.plant.critical {
    animation: plantCritical 1.8s ease-in-out infinite;
}

.plant.happy {
    animation: plantSway 3s ease-in-out infinite;
}

.plant-message {
    position: relative;
    z-index: 3;
    margin-top: 8px;
    font-size: 19px;
    font-weight: 700;
    color: #f8fafc;
}

.plant-action {
    position: relative;
    z-index: 3;
    margin-top: 5px;
    font-size: 14px;
    color: #cbd5e1;
}

@keyframes sunPulse {
    0%, 100% {
        transform: scale(1) rotate(0deg);
    }

    50% {
        transform: scale(1.12) rotate(8deg);
    }
}

@keyframes cloudMove {
    0%, 100% {
        transform: translateX(0);
    }

    50% {
        transform: translateX(35px);
    }
}

@keyframes rainFall {
    0% {
        transform: translateY(-35px);
    }

    100% {
        transform: translateY(35px);
    }
}

@keyframes plantSway {
    0%, 100% {
        transform: rotate(-2deg);
    }

    50% {
        transform: rotate(2deg);
    }
}

@keyframes plantDroop {
    0%, 100% {
        transform: rotate(-7deg) translateY(3px);
    }

    50% {
        transform: rotate(-13deg) translateY(8px);
    }
}

@keyframes plantCritical {
    0%, 100% {
        transform: rotate(-12deg) scale(0.96);
    }

    50% {
        transform: rotate(-18deg) scale(0.92);
    }
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    features = joblib.load(FEATURE_PATH)

    return model, features


try:

    model, features = load_model()

    model_ready = True
    model_error = ""

except Exception as e:

    model = None
    features = None
    model_ready = False
    model_error = str(e)


# ============================================================
# LOAD PLANT PROFILES FROM THE TRAINING DATASET
# ============================================================

@st.cache_data
def load_plant_profiles():
    dataset = pd.read_csv(DATASET_PATH)

    required_columns = [
        "Common_Name",
        "Scientific_Name",
        "Local_Name",
        "Crop_Category",
        "Indian_State",
        "Agro_Climate_Zone",
        "Optimal_Temperature_Min",
        "Optimal_Temperature_Max",
        "Optimal_Soil_Moisture_Min",
        "Optimal_Soil_Moisture_Max"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            "Training dataset is missing columns: "
            + ", ".join(missing_columns)
        )

    # Every Common_Name in the CSV becomes a selectable plant.
    profiles = (
        dataset.groupby("Common_Name", as_index=False)
        .agg({
            "Scientific_Name": "first",
            "Local_Name": "first",
            "Crop_Category": "first",
            "Indian_State": "first",
            "Agro_Climate_Zone": "first",
            "Optimal_Temperature_Min": "first",
            "Optimal_Temperature_Max": "first",
            "Optimal_Soil_Moisture_Min": "first",
            "Optimal_Soil_Moisture_Max": "first"
        })
        .sort_values("Common_Name")
        .reset_index(drop=True)
    )

    return profiles


try:

    plant_profiles = load_plant_profiles()
    plant_options = plant_profiles["Common_Name"].tolist()
    plant_data_ready = True
    plant_data_error = ""

except Exception as e:

    plant_profiles = pd.DataFrame()
    plant_options = ["Tomato"]
    plant_data_ready = False
    plant_data_error = str(e)


def get_selected_plant_profile(plant_name):

    if not plant_profiles.empty:

        selected_rows = plant_profiles[
            plant_profiles["Common_Name"] == plant_name
        ]

        if not selected_rows.empty:
            return selected_rows.iloc[0]

    return None


# ============================================================
# CONNECT TO ARDUINO
# ============================================================

@st.cache_resource
def connect_arduino():

    try:

        arduino = serial.Serial(
            ARDUINO_PORT,
            BAUD_RATE,
            timeout=0
        )

        time.sleep(2)

        arduino.reset_input_buffer()

        return arduino

    except Exception:

        return None


arduino = connect_arduino()


# ============================================================
# FAST NON-BLOCKING SENSOR READER
# ============================================================

def read_sensor_data():

    if arduino is None:
        return None

    latest_data = None

    try:

        while arduino.in_waiting > 0:

            line = arduino.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not line:
                continue

            # Expected Arduino format:
            #
            # DATA,soil,humidity,light,pump

            if line.startswith("DATA,"):

                parts = line.split(",")

                if len(parts) != 5:
                    continue

                try:

                    soil = float(parts[1])
                    humidity = float(parts[2])
                    light = float(parts[3])
                    pump = int(parts[4])

                    latest_data = {
                        "soil": soil,
                        "humidity": humidity,
                        "light": light,
                        "pump": pump
                    }

                except ValueError:

                    continue

        return latest_data

    except Exception:

        return None


# ============================================================
# GET CURRENT SENSOR DATA
# ============================================================

new_sensor_data = read_sensor_data()


# ============================================================
# KEEP LAST SENSOR READING
# ============================================================

if "last_sensor_data" not in st.session_state:

    st.session_state.last_sensor_data = {
        "soil": 0,
        "humidity": 0,
        "light": 0,
        "pump": 0
    }


if new_sensor_data is not None:

    st.session_state.last_sensor_data = new_sensor_data


sensor_data = st.session_state.last_sensor_data


soil_moisture = sensor_data["soil"]
humidity = sensor_data["humidity"]
light = sensor_data["light"]
pump = sensor_data["pump"]


# ============================================================
# RANDOM FOREST PREDICTION
# ============================================================

prediction = None
probabilities = None
confidence = 0


if model_ready:

    try:

        sensor_input = pd.DataFrame(
            [[
                soil_moisture,
                humidity,
                light
            ]],
            columns=features
        )

        prediction = int(
            model.predict(sensor_input)[0]
        )

        probabilities = model.predict_proba(
            sensor_input
        )[0]

        confidence = (
            max(probabilities) * 100
        )

    except Exception:

        prediction = None


# ============================================================
# STRESS LEVEL MEANING
# ============================================================

stress_names = {
    0: "HEALTHY",
    1: "MILD STRESS",
    2: "HIGH STRESS"
}

stress_emojis = {
    0: "🟢",
    1: "🟡",
    2: "🔴"
}


if prediction is not None:

    stress_name = stress_names.get(
        prediction,
        "UNKNOWN"
    )

    stress_emoji = stress_emojis.get(
        prediction,
        "⚠️"
    )

    stress_score = (
        probabilities[1] * 50
        + probabilities[2] * 100
    )

    stress_score = int(
        round(stress_score)
    )

else:

    stress_name = "MODEL ERROR"

    stress_emoji = "⚠️"

    stress_score = 0


# ============================================================
# LIVE WEATHER
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def fetch_live_weather():

    return get_live_weather()


try:

    weather_location, weather_data = (
        fetch_live_weather()
    )

    weather_available = True

except Exception as e:

    weather_location = None

    weather_data = (
        f"Weather unavailable: {e}"
    )

    weather_available = False


# ============================================================
# WEATHER PROCESSING
# ============================================================

weather_lower = (
    weather_data.lower()
    if isinstance(weather_data, str)
    else str(weather_data).lower()
)


rain = 1 if any(
    word in weather_lower
    for word in [
        "rain",
        "rainfall",
        "showers",
        "precipitation"
    ]
) else 0


# ============================================================
# WEATHER ANIMATION
# ============================================================

if rain:

    weather_class = "rainy"

    weather_icon = "🌧️"

    weather_effect = "rain"

    weather_text = (
        "Rain is expected or occurring!"
    )

elif humidity < 45:

    weather_class = "cloudy"

    weather_icon = "☁️"

    weather_effect = "cloud"

    weather_text = (
        "Dry atmospheric conditions"
    )

else:

    weather_class = "sunny"

    weather_icon = "☀️"

    weather_effect = "sun"

    weather_text = (
        "Pleasant conditions"
    )


# ============================================================
# PLANT-SPECIFIC THRESHOLDS FROM THE TRAINING DATASET
# ============================================================
# The plant is selected later in the sidebar, so use safe defaults here.
# They are replaced with the selected plant's CSV values after selection.
plant_soil_min = 30.0
plant_soil_max = 70.0
plant_temp_min = 15.0
plant_temp_max = 35.0

# ============================================================
# PLANT PERSONALITY
# ============================================================

if soil_moisture < plant_soil_min:

    plant_state = "thirsty"

    plant_emoji = "🥀"

    plant_message = (
        f"I'm thirsty! My optimal soil moisture starts at "
        f"{plant_soil_min:.0f}%."
    )

    plant_action = (
        f"Current soil moisture is below the plant "
        f"optimal range of {plant_soil_min:.0f}%–{plant_soil_max:.0f}%."
    )

elif soil_moisture > plant_soil_max:

    plant_state = "stressed"

    plant_emoji = "🌧️🥀"

    plant_message = (
        f"My soil is too wet for {plant}. "
        f"My upper optimal limit is {plant_soil_max:.0f}%."
    )

    plant_action = (
        f"Avoid extra irrigation and inspect drainage. "
        f"Optimal range: {plant_soil_min:.0f}%–{plant_soil_max:.0f}%."
    )

elif humidity < 35:

    plant_state = "stressed"

    plant_emoji = "😟🌱"

    plant_message = (
        "The air is too dry. I need more humidity."
    )

    plant_action = (
        "Increase humidity if suitable for this plant."
    )

elif light > 90:

    plant_state = "heat"

    plant_emoji = "☀️🥵🌱"

    plant_message = (
        "The light is too intense. "
        "Some shade would help."
    )

    plant_action = (
        "Consider partial shade during peak sunlight."
    )

elif prediction == 2:

    plant_state = "critical"

    plant_emoji = "🥀"

    plant_message = (
        "I need help! My stress level is high."
    )

    plant_action = (
        "Check the recommended actions and readings."
    )

else:

    plant_state = "happy"

    plant_emoji = "😊🌱"

    plant_message = (
        "I'm happy and feeling healthy!"
    )

    plant_action = (
        "Conditions look comfortable. Keep monitoring."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌱 PlantPulse AI")

    st.caption(
        "Intelligent Plant & Environmental Stress Prediction"
    )

    st.divider()

    st.markdown("### 🌿 Plant Profile")

    # All 25 plant species present in the training CSV are shown.
    plant = st.selectbox(
        "Select plant",
        plant_options,
        index=(
            plant_options.index("Tomato")
            if "Tomato" in plant_options
            else 0
        )
    )

    selected_plant_profile = get_selected_plant_profile(plant)

    # Load the selected plant's trained CSV thresholds.
    if selected_plant_profile is not None:
        plant_soil_min = float(selected_plant_profile["Optimal_Soil_Moisture_Min"])
        plant_soil_max = float(selected_plant_profile["Optimal_Soil_Moisture_Max"])
        plant_temp_min = float(selected_plant_profile["Optimal_Temperature_Min"])
        plant_temp_max = float(selected_plant_profile["Optimal_Temperature_Max"])

    if selected_plant_profile is not None:

        st.caption(
            f"🌿 {selected_plant_profile['Scientific_Name']}"
        )

        st.caption(
            "💧 Soil range: "
            f"{selected_plant_profile['Optimal_Soil_Moisture_Min']:.0f}% – "
            f"{selected_plant_profile['Optimal_Soil_Moisture_Max']:.0f}%"
        )

        st.caption(
            "🌡️ Temperature range: "
            f"{selected_plant_profile['Optimal_Temperature_Min']:.0f}°C – "
            f"{selected_plant_profile['Optimal_Temperature_Max']:.0f}°C"
        )

    elif not plant_data_ready:

        st.warning(
            "Plant profile dataset unavailable: "
            + plant_data_error
        )

    st.divider()

    st.markdown("### ⚙️ System")

    if arduino is not None:

        st.success(
            "● Arduino Connected"
        )

    else:

        st.error(
            "● Arduino Disconnected"
        )

    if model_ready:

        st.success(
            "🤖 ML Model: Ready"
        )

    else:

        st.error(
            "🤖 ML Model: Error"
        )

    if weather_available:

        st.success(
            "🌦️ Weather: Live"
        )

    else:

        st.warning(
            "🌦️ Weather: Unavailable"
        )

    st.write(
        "📡 Sensor Stream: Live"
    )

    if plant_data_ready:

        st.caption(
            f"🌿 Plant profiles loaded: {len(plant_options)}"
        )

    st.divider()

    st.caption("HYBRID MODE")

    st.caption(
        "Arduino sensors + Random Forest + Live Weather"
    )


# ============================================================
# REFRESH PLANT PERSONALITY AFTER PLANT SELECTION
# ============================================================
if selected_plant_profile is not None:

    if soil_moisture < plant_soil_min:
        plant_state = "thirsty"
        plant_emoji = "🥀"
        plant_message = f"I'm thirsty! My optimal soil moisture starts at {plant_soil_min:.0f}%."
        plant_action = f"Current soil moisture is below the {plant} optimal range of {plant_soil_min:.0f}%–{plant_soil_max:.0f}%."

    elif soil_moisture > plant_soil_max:
        plant_state = "stressed"
        plant_emoji = "🌧️🥀"
        plant_message = f"My soil is too wet for {plant}. My upper optimal limit is {plant_soil_max:.0f}%."
        plant_action = f"Avoid extra irrigation and inspect drainage. Optimal range: {plant_soil_min:.0f}%–{plant_soil_max:.0f}%."

    elif humidity < 35:
        plant_state = "stressed"
        plant_emoji = "😟🌱"
        plant_message = "The air is too dry. I need more humidity."
        plant_action = "Increase humidity if suitable for this plant."

    elif light > 90:
        plant_state = "heat"
        plant_emoji = "☀️🥵🌱"
        plant_message = "The light is too intense. Some shade would help."
        plant_action = "Consider partial shade during peak sunlight."

    elif prediction == 2:
        plant_state = "critical"
        plant_emoji = "🥀"
        plant_message = "I need help! My stress level is high."
        plant_action = "Check the recommended actions and readings."

    else:
        plant_state = "happy"
        plant_emoji = "😊🌱"
        plant_message = "I'm happy and feeling healthy!"
        plant_action = "Conditions look comfortable. Keep monitoring."


# ============================================================
# PAGE NAVIGATION
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = "Dashboard"


with st.sidebar:

    st.markdown("### 🧭 Navigation")

    if st.button(
        "🏠 Dashboard",
        width="stretch"
    ):

        st.session_state.page = "Dashboard"

        st.rerun()


    if st.button(
        "📊 Trends",
        width="stretch"
    ):

        st.session_state.page = "Trends"

        st.rerun()


    if st.button(
        "🌿 Plant Suitability",
        width="stretch"
    ):

        st.session_state.page = "Plant Suitability"

        st.rerun()


    if st.button(
        "🔮 What-If Simulation",
        width="stretch"
    ):

        st.session_state.page = "What-If Simulation"

        st.rerun()


    if st.button(
        "💬 AI Plant Assistant",
        width="stretch"
    ):

        st.session_state.page = "AI Plant Assistant"

        st.rerun()


    st.divider()

    st.caption(
        f"Current page: {st.session_state.page}"
    )


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    # ========================================================
    # HEADER
    # ========================================================

    st.html(
        """
        <div class="main-title">
            🌱 PlantPulse AI
        </div>

        <div class="subtitle">
            Intelligent Plant & Environmental Stress Prediction System
        </div>
        """
    )

    st.write("")


    # ========================================================
    # STATUS BAR
    # ========================================================

    col1, col2, col3 = st.columns(
        [2, 2, 2]
    )


    with col1:

        if arduino is not None:

            st.success(
                "🟢 LIVE MONITORING"
            )

        else:

            st.error(
                "🔴 SENSOR DISCONNECTED"
            )


    with col2:

        st.info(
            f"🌿 Plant: {plant}"
        )


    with col3:

        if prediction is not None:

            st.info(
                f"{stress_emoji} {stress_name}"
            )

        else:

            st.warning(
                "⚠️ Prediction unavailable"
            )


    # ========================================================
    # ENVIRONMENTAL CONDITIONS
    # ========================================================

    st.html(
        """
        <div class="section-title">
            🌡️ Environmental Conditions
        </div>
        """
    )


    # ========================================================
    # ENVIRONMENT ANIMATION
    # ========================================================

    if weather_effect == "rain":

        effect_html = (
            "🌧️ &nbsp; 🌧️ &nbsp; 🌧️ &nbsp; "
            "🌧️ &nbsp; 🌧️ &nbsp; 🌧️"
        )

    elif weather_effect == "cloud":

        effect_html = "☁️"

    else:

        effect_html = ""


    scene = f"""
    <div class="environment-scene {weather_class}">

        <div class="sun">
            ☀️
        </div>

        <div class="cloud">
            ☁️
        </div>

        <div class="rain">
            {effect_html if weather_effect == "rain" else ""}
        </div>

        <div class="plant {plant_state}">
            {plant_emoji}
        </div>

        <div class="plant-message">
            {plant_message}
        </div>

        <div class="plant-action">
            {plant_action}
        </div>

    </div>
    """

    st.html(scene)


    # ========================================================
    # LIVE SENSOR CARDS
    # ========================================================

    c1, c2, c3, c4, c5 = st.columns(5)


    # ========================================================
    # SOIL MOISTURE
    # ========================================================

    with c1:

        soil_status = (
            "Below Optimal"
            if soil_moisture < plant_soil_min
            else "Optimal"
            if soil_moisture <= plant_soil_max
            else "Above Optimal"
        )

        soil_card = f"""
        <div class="metric-card">

            <div class="metric-title">
                🌱 Soil Moisture
            </div>

            <div class="metric-value">
                {soil_moisture:.1f}%
            </div>

            <div class="metric-sub">
                {soil_status}
            </div>

        </div>
        """

        st.html(soil_card)


    # ========================================================
    # HUMIDITY
    # ========================================================

    with c2:

        humidity_status = (
            "Low"
            if humidity < 40
            else "Normal"
            if humidity < 80
            else "High"
        )

        humidity_card = f"""
        <div class="metric-card">

            <div class="metric-title">
                💧 Humidity
            </div>

            <div class="metric-value">
                {humidity:.1f}%
            </div>

            <div class="metric-sub">
                {humidity_status}
            </div>

        </div>
        """

        st.html(humidity_card)


    # ========================================================
    # LIGHT
    # ========================================================

    with c3:

        light_status = (
            "Low"
            if light < 30
            else "Normal"
            if light < 75
            else "High"
        )

        light_card = f"""
        <div class="metric-card">

            <div class="metric-title">
                ☀️ Light Level
            </div>

            <div class="metric-value">
                {light:.1f}%
            </div>

            <div class="metric-sub">
                {light_status}
            </div>

        </div>
        """

        st.html(light_card)


    # ========================================================
    # WATER PUMP
    # ========================================================

    with c4:

        pump_status = (
            "ON"
            if pump == 1
            else "OFF"
        )

        pump_card = f"""
        <div class="metric-card">

            <div class="metric-title">
                🔌 Water Pump
            </div>

            <div class="metric-value">
                {pump_status}
            </div>

            <div class="metric-sub">
                Arduino relay status
            </div>

        </div>
        """

        st.html(pump_card)


    # ========================================================
    # RAINFALL
    # ========================================================

    with c5:

        rainfall_card = f"""
        <div class="metric-card">

            <div class="metric-title">
                🌧️ Rainfall
            </div>

            <div class="metric-value">
                {"Yes" if rain else "No"}
            </div>

            <div class="metric-sub">
                Live weather
            </div>

        </div>
        """

        st.html(rainfall_card)


    # ========================================================
    # LIVE WEATHER DETAILS
    # ========================================================

    st.html(
        """
        <div class="section-title">
            🌦️ Live Weather & Rain Forecast
        </div>
        """
    )


    weather_col1, weather_col2 = st.columns(
        [1, 2]
    )


    with weather_col1:

        if (
            weather_available
            and weather_location
        ):

            city = weather_location.get(
                "city",
                "Unknown"
            )

            state = weather_location.get(
                "state",
                ""
            )

            country = weather_location.get(
                "country",
                ""
            )

            location_card = f"""
            <div class="metric-card">

                <div class="metric-title">
                    📍 Auto-Detected Location
                </div>

                <div class="metric-value"
                     style="font-size:22px;">

                    {city}

                </div>

                <div class="metric-sub">

                    {state}, {country}

                </div>

            </div>
            """

            st.html(location_card)

        else:

            st.warning(
                "📍 Live location/weather unavailable"
            )


    with weather_col2:

        st.info(
            weather_data
        )


    st.write("")


    # ========================================================
    # AI PLANT HEALTH ANALYSIS
    # ========================================================

    st.html(
        """
        <div class="section-title">
            🧠 AI Plant Health Analysis
        </div>
        """
    )


    left, right = st.columns(
        [1, 2]
    )


    # ========================================================
    # STRESS SCORE
    # ========================================================

    with left:

        if prediction == 0:

            label_color = "#22c55e"

        elif prediction == 1:

            label_color = "#f59e0b"

        else:

            label_color = "#ef4444"


        stress_card = f"""
        <div class="stress-box">

            <div style="color:#64748b;">
                Plant Stress Score
            </div>

            <div class="stress-score">

                {stress_score}

                <span style="font-size:25px;">
                    /100
                </span>

            </div>

            <div class="stress-label"
                 style="color:{label_color};">

                {stress_emoji}
                {stress_name}

            </div>

            <div style="
                color:#94a3b8;
                margin-top:10px;
                font-size:14px;
            ">

                Model Confidence:
                {confidence:.1f}%

            </div>

        </div>
        """

        st.html(stress_card)


    # ========================================================
    # STRESS FACTORS
    # ========================================================

    with right:

        st.markdown(
            "#### 🔍 Likely Stress Factors"
        )


        factor_values = {

            "Low soil moisture":
                max(
                    0,
                    100 - soil_moisture
                ),

            "Low humidity":
                max(
                    0,
                    60 - humidity
                ),

            "Excessive light":
                max(
                    0,
                    light - 60
                )
        }


        factors = pd.DataFrame({

            "Factor":
                list(
                    factor_values.keys()
                ),

            "Contribution":
                list(
                    factor_values.values()
                )

        })


        factors = factors[
            factors["Contribution"] > 0
        ]


        if factors.empty:

            st.success(
                "🟢 No major environmental stress factor detected."
            )

        else:

            factors["Contribution"] = (

                factors["Contribution"]
                /
                factors["Contribution"].sum()
                *
                100

            )


            factors = factors.sort_values(
                "Contribution",
                ascending=True
            )


            fig = go.Figure(

                go.Bar(

                    x=factors["Contribution"],

                    y=factors["Factor"],

                    orientation="h"

                )

            )


            fig.update_layout(

                height=230,

                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=10
                ),

                xaxis_title=
                    "Relative Contribution (%)",

                yaxis_title="",

                showlegend=False,

                paper_bgcolor="#111827",

                plot_bgcolor="#111827",

                font=dict(
                    color="#e5e7eb"
                )
            )


            st.plotly_chart(
                fig,
                width="stretch"
            )


    st.caption(
        f"{weather_icon} Environment: "
        f"{weather_text}  •  "
        f"🌱 Plant state: "
        f"{plant_state.replace('_', ' ').title()}"
    )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    st.html(
        """
        <div class="section-title">
            💡 Recommended Action
        </div>
        """
    )


    recommendations = []


    if soil_moisture < plant_soil_min:

        recommendations.append(
            "💧 <strong>Water the plant</strong> — "
            f"{plant} prefers soil moisture of "
            f"{plant_soil_min:.0f}%–{plant_soil_max:.0f}%, "
            f"but the current reading is {soil_moisture:.1f}%."
        )

    elif soil_moisture <= plant_soil_max:

        recommendations.append(
            "🟢 <strong>Soil moisture is within the "
            f"{plant} optimal range</strong> — "
            f"{plant_soil_min:.0f}%–{plant_soil_max:.0f}%."
        )

    else:

        recommendations.append(
            "⚠️ <strong>Soil moisture is above the "
            f"{plant} optimal range</strong> — "
            f"reduce irrigation and check drainage."
        )


    if humidity < 40:

        recommendations.append(
            "💦 <strong>Increase humidity</strong> if the "
            "selected plant prefers humid conditions."
        )


    if light > 80:

        recommendations.append(
            "☀️ <strong>Provide partial shade</strong> — "
            "light level is currently high."
        )


    if prediction == 0:

        recommendations.append(
            "🟢 <strong>Conditions look healthy.</strong> "
            "Continue monitoring."
        )

    elif prediction == 1:

        recommendations.append(
            "🟡 <strong>Monitor the plant closely</strong> "
            "and correct the environmental factors above."
        )

    elif prediction == 2:

        recommendations.append(
            "🚨 <strong>Immediate attention recommended</strong> "
            "because the ML model predicts high stress."
        )


    if not recommendations:

        recommendations.append(
            "🟢 <strong>No immediate action required.</strong> "
            "Continue monitoring."
        )


    recommendation_html = (
        "<br><br>".join(
            recommendations
        )
    )


    recommendation_card = f"""
    <div class="recommendation">

        {recommendation_html}

    </div>
    """

    st.html(recommendation_card)


# ============================================================
# TRENDS
# ============================================================

elif st.session_state.page == "Trends":

    st.subheader(
        "📈 Environmental Trends"
    )


    if "history" not in st.session_state:

        st.session_state.history = []


    if new_sensor_data is not None:

        st.session_state.history.append({

            "Time":
                pd.Timestamp.now(),

            "Humidity":
                humidity,

            "Soil Moisture":
                soil_moisture,

            "Light":
                light

        })


    st.session_state.history = (
        st.session_state.history[-50:]
    )


    history_df = pd.DataFrame(
        st.session_state.history
    )


    selected_parameter = st.selectbox(

        "Select parameter",

        [
            "Soil Moisture",
            "Humidity",
            "Light"
        ]

    )


    if not history_df.empty:

        fig = go.Figure()


        fig.add_trace(

            go.Scatter(

                x=history_df["Time"],

                y=history_df[
                    selected_parameter
                ],

                mode="lines+markers",

                name=selected_parameter

            )

        )


        fig.update_layout(

            height=400,

            xaxis_title="Time",

            yaxis_title=
                selected_parameter,

            hovermode="x unified",

            paper_bgcolor="#111827",

            plot_bgcolor="#111827",

            font=dict(
                color="#e5e7eb"
            )

        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    else:

        st.info(
            "Waiting for Arduino sensor readings..."
        )


    st.caption(
        "Live trend data is collected from Arduino."
    )


# ============================================================
# PLANT SUITABILITY
# ============================================================

elif st.session_state.page == "Plant Suitability":

    st.subheader(
        "🌿 Plant Suitability"
    )


    st.write(
        "Based on the current environmental conditions, "
        "these plants have different suitability levels."
    )


    suitability_rows = []

    for _, profile in plant_profiles.iterrows():

        minimum = float(
            profile["Optimal_Soil_Moisture_Min"]
        )

        maximum = float(
            profile["Optimal_Soil_Moisture_Max"]
        )

        if minimum <= soil_moisture <= maximum:

            score = 100.0

        elif soil_moisture < minimum:

            distance = minimum - soil_moisture
            score = max(
                0.0,
                100.0 - (distance / max(minimum, 1.0) * 100.0)
            )

        else:

            distance = soil_moisture - maximum
            score = max(
                0.0,
                100.0 - (distance / max(100.0 - maximum, 1.0) * 100.0)
            )

        suitability_rows.append({
            "Plant": profile["Common_Name"],
            "Suitability": round(score, 1)
        })

    suitability = pd.DataFrame(
        suitability_rows
    ).sort_values(
        "Suitability",
        ascending=False
    )


    fig = go.Figure(

        go.Bar(

            x=suitability["Plant"],

            y=suitability["Suitability"],

            text=suitability[
                "Suitability"
            ].apply(
                lambda x: f"{x}%"
            ),

            textposition="outside"

        )

    )


    fig.update_layout(

        height=380,

        yaxis=dict(
            range=[0, 105]
        ),

        yaxis_title=
            "Suitability (%)",

        xaxis_title=
            "Plant",

        showlegend=False,

        paper_bgcolor="#111827",

        plot_bgcolor="#111827",

        font=dict(
            color="#e5e7eb"
        )

    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


    st.info(
        "Suitability is calculated dynamically for every plant "
        "in the 75,000-row training dataset using its "
        "plant-specific optimal soil-moisture range."
    )


# ============================================================
# WHAT-IF SIMULATION
# ============================================================

elif st.session_state.page == "What-If Simulation":

    st.subheader(
        "🔮 What-If Simulation"
    )


    st.write(
        "Experiment with environmental conditions and "
        "see how the trained Random Forest model "
        "could classify the resulting stress level."
    )


    if selected_plant_profile is not None:

        st.info(
            f"🌿 {plant} optimal soil moisture: "
            f"{plant_soil_min:.0f}%–{plant_soil_max:.0f}%  •  "
            f"🌡️ Optimal temperature: "
            f"{plant_temp_min:.0f}°C–{plant_temp_max:.0f}°C"
        )


    col1, col2 = st.columns(2)


    with col1:

        simulated_moisture = st.slider(

            "💧 Soil Moisture",

            0,

            100,

            int(soil_moisture)

        )


        simulated_humidity = st.slider(

            "💦 Humidity",

            0,

            100,

            int(humidity)

        )


    with col2:

        simulated_light = st.slider(

            "☀️ Light Level",

            0,

            100,

            int(light)

        )


    if model_ready:

        simulation_data = pd.DataFrame(

            [[
                simulated_moisture,
                simulated_humidity,
                simulated_light
            ]],

            columns=features

        )


        try:

            simulated_prediction = (
                model.predict(
                    simulation_data
                )[0]
            )


            simulated_probabilities = (
                model.predict_proba(
                    simulation_data
                )[0]
            )


            simulated_confidence = (
                max(
                    simulated_probabilities
                ) * 100
            )


            st.divider()


            st.metric(

                "Simulated Stress Level",

                stress_names[
                    int(
                        simulated_prediction
                    )
                ]

            )


            st.metric(

                "Model Confidence",

                f"{simulated_confidence:.1f}%"

            )


            if simulated_prediction == 0:

                st.success(
                    "🟢 Simulated condition: HEALTHY"
                )

            elif simulated_prediction == 1:

                st.warning(
                    "🟡 Simulated condition: MILD STRESS"
                )

            else:

                st.error(
                    "🔴 Simulated condition: HIGH STRESS"
                )


        except Exception as e:

            st.error(
                f"Simulation error: {e}"
            )


    else:

        st.error(
            "Random Forest model is unavailable."
        )


# ============================================================
# AI PLANT ASSISTANT
# ============================================================

elif st.session_state.page == "AI Plant Assistant":

    st.subheader(
        "💬 AI Plant Assistant"
    )


    st.write(
        "Ask questions about your plant, environmental "
        "conditions, or upload a plant image for analysis."
    )


    uploaded_image = st.file_uploader(

        "📷 Upload a plant/leaf image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]

    )

    if uploaded_image is not None:

        import base64

        image_bytes = uploaded_image.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode()

        # Keep the uploaded image small without changing Streamlit's image width API.
        st.markdown(
            f"""
            <div style="text-align:center;">
                <img
                    src="data:{uploaded_image.type};base64,{image_base64}"
                    style="
                        width:250px;
                        height:250px;
                        object-fit:cover;
                        border-radius:12px;
                    "
                >
                <p style="color:#cbd5e1;">
                    Uploaded Plant Image
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### 🔍 AI Image Analysis")

        # The image is sent to Gemini Vision together with the selected
        # plant and live PlantPulse context. No fixed confidence value is used.
        image_context = build_plant_context(
            plant=plant,
            soil_moisture=soil_moisture,
            humidity=humidity,
            light=light,
            stress_level=stress_name,
            stress_score=stress_score,
            rainfall="Yes" if rain else "No",
            soil_min=plant_soil_min,
            soil_max=plant_soil_max,
            temp_min=plant_temp_min,
            temp_max=plant_temp_max,
            temperature=None
        )
        image_context += f"\nLive Weather Information:\n{weather_data}\n"

        if st.button(
            "🌿 Analyze Uploaded Image",
            key="analyze_plant_image_button",
            width="stretch"
        ):
            with st.spinner("🔎 PlantPulse AI is analyzing the image..."):
                try:
                    image_analysis = analyze_plant_image(
                        image_bytes=image_bytes,
                        mime_type=uploaded_image.type,
                        context=image_context
                    )

                    st.session_state.plantpulse_image_analysis = image_analysis

                except Exception as e:
                    st.error(
                        f"Unable to analyze the plant image: {e}"
                    )

        if "plantpulse_image_analysis" in st.session_state:
            st.success("✅ Image analysis completed")
            st.markdown(
                st.session_state.plantpulse_image_analysis
            )
            st.caption(
                "Note: Visual analysis is an AI assessment of visible features. "
                "It is not a laboratory diagnosis, and no artificial confidence "
                "percentage is displayed."
            )

    st.divider()


    # Build the current PlantPulse context for Gemini.
    # Temperature is not available from the current Arduino data stream,
    # so it is intentionally passed as unavailable rather than invented.
    plant_context = build_plant_context(
        plant=plant,
        soil_moisture=soil_moisture,
        humidity=humidity,
        light=light,
        stress_level=stress_name,
        stress_score=stress_score,
        rainfall="Yes" if rain else "No",
        soil_min=plant_soil_min,
        soil_max=plant_soil_max,
        temp_min=plant_temp_min,
        temp_max=plant_temp_max,
        temperature=None
    )

    # Keep the existing weather information available to the chatbot context.
    plant_context += f"\nLive Weather Information:\n{weather_data}\n"


    # Store conversation history so the AI chat remains visible after
    # Streamlit reruns caused by the live sensor refresh.
    if "plantpulse_chat" not in st.session_state:

        st.session_state.plantpulse_chat = []


    # Display previous chat messages.
    for message in st.session_state.plantpulse_chat:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


    question = st.chat_input(

        f"Ask PlantPulse AI about your {plant}..."

    )


    if question:

        with st.chat_message("user"):

            st.markdown(question)


        st.session_state.plantpulse_chat.append({
            "role": "user",
            "content": question
        })


        with st.chat_message("assistant"):

            with st.spinner("🌱 PlantPulse AI is thinking..."):

                try:

                    answer = ask_gemini(
                        question,
                        plant_context
                    )

                    st.markdown(answer)

                    st.session_state.plantpulse_chat.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:

                    st.error(
                        f"Unable to contact PlantPulse AI: {e}"
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "🌱 PlantPulse AI | Sense → Predict → Explain → Act"
)


# ============================================================
# FAST AUTO REFRESH
# ============================================================

# Streamlit checks the Arduino buffer every second.
#
# Arduino itself controls how frequently DATA is generated.
# With delay(5000), Arduino sends a new reading every 5 sec.
#
# Streamlit refreshes every 1 second so there is no additional
# long waiting period.

time.sleep(1)

st.rerun()