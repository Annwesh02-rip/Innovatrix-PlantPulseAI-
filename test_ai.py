from ai_assistant.chatbot import ask_gemini

context = """
Selected Plant: Tomato

Current Sensor Readings:
- Soil Moisture: 22%
- Humidity: 48%
- Light Intensity: 650
- Temperature: 29°C
- Rainfall: No

Plant Stress:
- Stress Level: MILD STRESS
- Stress Score: 42%

Plant Optimal Conditions:
- Soil Moisture: 30% to 70%
- Temperature: 20°C to 30°C
"""

question = "Why is my tomato plant stressed?"

answer = ask_gemini(question, context)

print("\nAI RESPONSE:\n")
print(answer)