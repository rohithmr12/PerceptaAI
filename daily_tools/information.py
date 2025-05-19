# daily_tools/information.py
import os
import requests
from typing import List, Dict, Any, Optional

# --- Weather Functionality (using Open-Meteo) ---
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

def get_coordinates_for_city(city_name: str) -> Optional[Dict[str, float]]:
    """Fetches latitude and longitude for a given city name using Open-Meteo Geocoding."""
    params = {"name": city_name, "count": 1, "language": "en", "format": "json"}
    try:
        response = requests.get(GEOCODING_API_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("results") and len(data["results"]) > 0:
            location = data["results"][0]
            return {"latitude": location["latitude"], "longitude": location["longitude"], "name": location.get("name", city_name)}
        else:
            print(f"Geocoding: Could not find coordinates for {city_name}.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Geocoding API error for {city_name}: {e}")
        return None

def get_weather_actual(city: str) -> str:
    """Fetches actual weather information for a given city using Open-Meteo."""
    coordinates = get_coordinates_for_city(city)
    if not coordinates:
        return f"Could not find location information for {city} to get weather."

    params = {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "current_weather": "true"
    }
    try:
        response = requests.get(WEATHER_API_URL, params=params, timeout=5)
        response.raise_for_status()
        weather_data = response.json()
        
        current_weather = weather_data.get("current_weather")
        if current_weather:
            temp = current_weather.get("temperature")
            windspeed = current_weather.get("windspeed")
            # Weather code interpretation would be nice, but Open-Meteo doesn't provide text directly.
            # For simplicity, we'll just return temp and wind.
            return f"Weather in {coordinates.get('name', city)}: Temperature {temp}°C, Windspeed {windspeed} km/h."
        else:
            return f"Could not retrieve current weather details for {coordinates.get('name', city)}."
            
    except requests.exceptions.RequestException as e:
        print(f"Weather API error for {coordinates.get('name', city)}: {e}")
        return f"Error fetching weather for {coordinates.get('name', city)}: {e}"
    except KeyError as e:
        print(f"Weather API response format error for {coordinates.get('name', city)}: {e}")
        return f"Error parsing weather data for {coordinates.get('name', city)}."

# --- News Functionality (using NewsAPI.org) ---
NEWS_API_BASE_URL = "https://newsapi.org/v2/top-headlines"
# NEWS_API_KEY should be set as an environment variable

def get_news_actual(category: str = "general", country: str = "us") -> List[str]:
    """Fetches actual news headlines from NewsAPI.org. Requires NEWS_API_KEY environment variable."""
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return ["NewsAPI key (NEWS_API_KEY) is not set. Please configure it to fetch news."]

    params = {
        "apiKey": api_key,
        "category": category.lower() if category else "general",
        "country": country.lower(),
        "pageSize": 5 # Limit number of headlines
    }
    
    try:
        response = requests.get(NEWS_API_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        news_data = response.json()
        
        articles = news_data.get("articles")
        if articles:
            headlines = [article.get("title", "No title") for article in articles]
            if not headlines:
                return [f"No news articles found for category '{category}' in country '{country}'."]
            return headlines
        else:
            status = news_data.get("status")
            if status == "error":
                return [f"NewsAPI Error: {news_data.get('message', 'Unknown error from API.')}"]
            return [f"Could not retrieve news articles for category '{category}' in country '{country}'."]
            
    except requests.exceptions.RequestException as e:
        print(f"NewsAPI request error for category {category}: {e}")
        return [f"Error fetching news for category '{category}': {e}"]
    except KeyError as e:
        print(f"NewsAPI response format error for category {category}: {e}")
        return [f"Error parsing news data for category '{category}'."]

# --- Mock functions (can be removed or kept for fallback/testing) ---
def get_weather_mock(city: str) -> str:
    """Mocks fetching weather information."""
    print(f"MOCK: Getting weather for {city}")
    # Simulate different weather for a few cities for variety
    if city.lower() == "london":
        return f"Mock: The weather in {city} is cloudy and 15°C."
    elif city.lower() == "tokyo":
        return f"Mock: The weather in {city} is rainy and 20°C."
    return f"Mock: The weather in {city} is sunny and 25°C."

def get_news_mock(category: str = "general") -> List[str]:
    """Mocks fetching news headlines."""
    print(f"MOCK: Getting news for category: {category}")
    return [
        f"Mock News Headline 1 for {category} - Exciting developments!",
        f"Mock News Headline 2 for {category} - Market trends analyzed.",
        f"Mock News Headline 3 for {category} - Local events upcoming."
    ] 