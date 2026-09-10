# ============================================================
#              PLANTPULSE AI
#       REAL-TIME SENSOR PREDICTION
#       Arduino + PySerial + Random Forest
#       + Live Weather Integration
# ============================================================

import serial
import time
import joblib
import pandas as pd

from weather_api import get_live_weather


# ============================================================
# 1. SETTINGS
# ============================================================

PORT = "COM12"          # Arduino COM port
BAUD_RATE = 9600


# ============================================================
# 2. LOAD RANDOM FOREST MODEL
# ============================================================

print("Loading Random Forest model...")

model = joblib.load("random_forest_model.pkl")

print("Random Forest model loaded successfully!")


# ============================================================
# 3. LOAD FEATURE INFORMATION
# ============================================================

features = joblib.load("model_features.pkl")

print("\nModel features:")
print(features)


# ============================================================
# 4. CONNECT TO ARDUINO
# ============================================================

print("\nConnecting to Arduino...")

arduino = serial.Serial(
    PORT,
    BAUD_RATE,
    timeout=2
)

time.sleep(2)

# Remove old/stale serial data
arduino.reset_input_buffer()

print("Arduino connected!")
print("Waiting for real-time sensor data...\n")


# ============================================================
# 5. GET LIVE WEATHER
# ============================================================

print("Fetching live weather information...")

try:

    location, weather = get_live_weather()

    print("\n========================================")
    print("        LIVE WEATHER INFORMATION")
    print("========================================")

    print(
        f"Location     : {location['city']}, "
        f"{location['state']}, "
        f"{location['country']}"
    )

    print(
        f"Coordinates  : "
        f"{location['latitude']}, "
        f"{location['longitude']}"
    )

    print("\nWeather:")
    print(weather)

    print("========================================\n")


except Exception as e:

    print("\nWeather API could not be accessed.")
    print("Error:", e)

    location = None
    weather = "Weather information unavailable."


# ============================================================
# 6. STRESS LEVEL MEANING
# ============================================================

stress_names = {
    0: "HEALTHY",
    1: "MILD STRESS",
    2: "HIGH STRESS"
}


# ============================================================
# 7. READ REAL-TIME DATA
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # Read Arduino serial line
        # ----------------------------------------------------

        line = arduino.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()


        # Ignore empty lines
        if not line:
            continue


        # ----------------------------------------------------
        # Only process machine-readable DATA lines
        # ----------------------------------------------------

        if not line.startswith("DATA,"):
            continue


        # ----------------------------------------------------
        # DATA FORMAT:
        #
        # DATA,soil,humidity,light,pump
        #
        # Example:
        #
        # DATA,42,65.5,78,0
        # ----------------------------------------------------

        parts = line.split(",")


        # Make sure there are exactly 5 values
        if len(parts) != 5:

            print("Invalid data:", line)

            continue


        # ====================================================
        # 8. EXTRACT SENSOR VALUES
        # ====================================================

        try:

            soil = float(parts[1])

            humidity = float(parts[2])

            light = float(parts[3])

            pump = int(parts[4])


        except ValueError:

            print(
                "Could not read sensor values:",
                line
            )

            continue


        # ====================================================
        # 9. CREATE ML INPUT
        # ====================================================

        sensor_data = pd.DataFrame(
            [[
                soil,
                humidity,
                light
            ]],
            columns=features
        )


        # ====================================================
        # 10. RANDOM FOREST PREDICTION
        # ====================================================

        prediction = model.predict(
            sensor_data
        )[0]


        # ====================================================
        # 11. PREDICTION PROBABILITY
        # ====================================================

        probabilities = model.predict_proba(
            sensor_data
        )[0]

        confidence = max(probabilities) * 100


        # ====================================================
        # 12. DISPLAY REAL-TIME ANALYSIS
        # ====================================================

        print("\n========================================")
        print("          PLANTPULSE AI")
        print("       REAL-TIME ANALYSIS")
        print("========================================")

        print(
            f"Soil Moisture : {soil:.1f}%"
        )

        print(
            f"Humidity      : {humidity:.1f}%"
        )

        print(
            f"Light Level   : {light:.1f}%"
        )

        print(
            f"Pump          : "
            f"{'ON' if pump == 1 else 'OFF'}"
        )

        print("----------------------------------------")

        print(
            f"Stress Level  : {prediction}"
        )

        print(
            f"Status        : "
            f"{stress_names.get(prediction, 'UNKNOWN')}"
        )

        print(
            f"Confidence    : {confidence:.2f}%"
        )

        print("----------------------------------------")

        # ====================================================
        # 13. LIVE WEATHER
        # ====================================================

        print("LIVE WEATHER")

        if location:

            print(
                f"Location      : "
                f"{location['city']}, "
                f"{location['state']}"
            )

            print(
                f"Weather       : {weather}"
            )

        else:

            print(
                "Weather       : Unavailable"
            )

        print("========================================")


# ============================================================
# 14. STOP PROGRAM
# ============================================================

except KeyboardInterrupt:

    print("\nStopping PlantPulse...")

    arduino.close()

    print("Arduino connection closed.")