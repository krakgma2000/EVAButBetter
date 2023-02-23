from .action import Action
import requests


class Weather(Action):
    def __init__(self):
        self.weather_api_key = "5e5e253aee5dff74e6ea42b37e1fba33"
        self.city = "Barcelona"
        self.country = "ES"
        self.weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={self.city},{self.country}&appid={self.weather_api_key}"

    def run(self,action_obj=None):
        weather_data = requests.get(self.weather_url).json()
        weather = weather_data["weather"][0]["description"]
        temperature = weather_data["main"]["temp"] - 273.15
        temperature = round(temperature, 2)
        return f"The weather in {self.city} is {weather} and the temperature is {temperature}°C."