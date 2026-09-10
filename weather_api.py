import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def detect_location():
    """
    Automatically detects the user's approximate location
    using their public IP address.
    """

    url = "http://ip-api.com/json/"

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()

    if data.get("status") != "success":
        raise Exception("Unable to detect location.")

    return {
        "city": data.get("city"),
        "state": data.get("regionName"),
        "country": data.get("country"),
        "latitude": data.get("lat"),
        "longitude": data.get("lon")
    }


def get_live_weather():
    """
    Automatically detects location and asks OpenAI
    to fetch current weather and today's forecast.
    """

    location = detect_location()

    city = location["city"]
    state = location["state"]
    country = location["country"]

    prompt = f"""
    Get the latest real-time weather information and today's
    weather forecast for {city}, {state}, {country}.

    Specifically provide:

    1. Current temperature in Celsius
    2. Current humidity
    3. Current weather condition
    4. Rain probability
    5. Expected rainfall
    6. Whether rain is expected today
    7. Short weather forecast for today

    Keep the response concise and suitable for a
    plant-monitoring dashboard.

    IMPORTANT:
    Use current web information and do not guess the weather.
    """

    response = client.responses.create(
        model="gpt-5",
        tools=[
            {
                "type": "web_search"
            }
        ],
        input=prompt
    )

    return location, response.output_text


if __name__ == "__main__":

    print("📍 Detecting your location...")

    try:
        location, weather = get_live_weather()

        print("\n================================")
        print("       PLANTPULSE AI")
        print("       LIVE WEATHER")
        print("================================")

        print(f"\n📍 Location: {location['city']}, "
              f"{location['state']}, {location['country']}")

        print(f"🌐 Coordinates: "
              f"{location['latitude']}, {location['longitude']}")

        print("\n🌦️ Weather Information:")
        print(weather)

    except Exception as e:
        print("\n❌ Weather Error:")
        print(e)