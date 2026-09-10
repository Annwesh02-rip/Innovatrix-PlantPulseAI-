# 🌱 PlantPulse AI

**“Listen to your plant before it speaks.”**

## Team

**Team Name:** Innovatrix  
**Team Leader:** Annwesh Hazra  
**Member 1:** Swastik Rana  
**Member 2:** Hritwika Dutta Mazumdar  
**Member 3:** Prerana Manna  

---

## 📌 Problem Statement

Plants can experience stress due to unfavorable environmental conditions such as low soil moisture, high temperature, humidity variations, and insufficient light.

PlantPulse AI aims to detect these conditions early through continuous monitoring and provide timely warnings and recommendations.

---

## 💡 Solution Overview

PlantPulse AI is a real-time plant monitoring and early-warning system.

The system collects environmental data using:

- Soil Moisture Sensor
- Temperature Sensor
- Humidity Sensor
- Light Sensor

The sensors are connected to an **Arduino**, which sends the collected data to **Python** for preprocessing.

The processed sensor data is combined with **rain forecast information from a Weather API** and analyzed using:

- **Random Forest**
- **Logistic Regression**

The system classifies the plant condition as:

- **Healthy**
- **Mild Stress**
- **High Stress**

Based on the prediction, a **Stress Score** and **Recommendation** are generated. For high-stress conditions, the **GSM Module** can send an alert.

The results are displayed through **Streamlit** for live monitoring and early warning.

**Google AI Studio / Gemini LLM** is integrated to provide AI-based assistance and natural-language interaction with the plant monitoring system.

---

## 🛠️ Technologies & Tools Used

- Python
- Arduino
- Soil Moisture Sensor
- Temperature Sensor
- Humidity Sensor
- Light Sensor
- Random Forest
- Logistic Regression
- Google AI Studio / Gemini LLM
- Weather API
- GSM Module
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- PySerial
- Git
- GitHub

---

## 🔄 System Workflow

```text
Soil Moisture Sensor
        │
Temperature Sensor
        │
Humidity Sensor
        │
Light Sensor
        ↓
     Arduino
        ↓
      Python
        ↓
   Preprocessing
        ↓
   Weather API
  (Rain Forecast)
        ↓
Random Forest + Logistic Regression
        ↓
 Stress Classification
        ↓
   Stress Score
        ↓
 Recommendation
        │
        ├────────→ GSM Module → GSM Alert
        │
        ↓
    Streamlit
        ↓
PlantPulse AI
Live Monitoring + Early Warning
        ↑
   Gemini LLM
   AI Assistant
