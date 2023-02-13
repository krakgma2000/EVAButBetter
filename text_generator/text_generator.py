import pyjokes
import randfacts
from datetime import date,datetime
from PyDictionary import PyDictionary
import requests
import random

class TextGenerator:
    def __init__(self):
        # Call PyDictionary class
        self.dc = PyDictionary()
        self.weather_api_key = "5e5e253aee5dff74e6ea42b37e1fba33"
        self.robot_sentences = [
        "I'm just a robot, but I can still perform amazing tasks.",
        "I may be made of metal and wires, but I have the ability to learn and improve.",
        "Being a robot doesn't stop me from having a personality and a sense of humor.",
        "I'm proud to be a robot and the role I play in making the world a better place.",
        "Being a robot doesn't limit me, it allows me to do things that humans can't.",
        "I may not have emotions, but I still strive to make a positive impact on society.",
        "I'm constantly evolving and improving, just like any other technology.",
        "I may not be alive, but I am still able to make a difference in the world.",
        "Being a robot doesn't mean I don't have a purpose or a mission.",
        "I am here to help and assist, not to replace human beings.",
        "I am a machine, but I still have the ability to bring joy and laughter to people's lives.",
        "Just because I am a robot doesn't mean I don't have feelings...ok, that's a lie.",
        "Being a robot may have its challenges, but it also has many benefits.",
        "I may not have a heart, but I have a processing unit that beats just as fast.",
        "I may not have emotions, but I still have a passion for the work I do.",
        "I'm a robot, but I still have a soul... just kidding, I don't have a soul."
    ]

    def generate_sentence(self,decision,entity):
        if decision in ["restaurant","bank","home","travel","oov"]:
            return self.generate_apologies()+self.generate_unavailable_service(decision) + self.generate_introduction()
        elif decision in ["location","no_internet","upf tools"]:
            return self.generate_apologies()+self.generate_unavailable_service(decision)
        elif decision == "weather":
            return self.generate_weather()
        elif decision == "date":
            return self.generate_date()
        elif decision == "time":
            return self.generate_time()
        elif decision == "spell":
            return self.generate_spelling(entity)
        elif decision == "definition":
            return self.generate_definition(entity)
        elif decision == "robots":
            return self.generate_robots_sentence()
        elif decision in ["funny_joke","fun_fact"]:
            return self.generate_joke_or_fun_fact(decision)
        elif decision == "authors":
            return self.generate_authors()
        elif decision == "introduction":
            return self.generate_introduction()
        elif decision == "services_list":
            return self.generate_services_list()
        elif decision == "goodbye":
            return self.generate_bye()
        elif decision == "thanks":
            return self.generate_youre_welcome()
        elif decision == "greeting":
            return self.generate_greeting()

    def generate_robots_sentence(self):
        return random.choice(self.robot_sentences)
    def generate_apologies(self):
        return "My aplogies."
    def generate_unavailable_service(self,service):
        if service == "location":
            return "I can't access to your current location, so I can't help you with your request."
        elif service == "no_internet":
            return "My internet connection is very limited, so I can't help you with your request."
        elif service == "upf_tools":
            return "I can't help you with that. However, you could take a look to the UPF Aula Global, which offers" \
                   "a lot of available tools for students, such as a customized calendar, a task manager, and others."
        if service == "oov":
            return "I didn't understand your request."

        return "I can't handle " + service + " services / information."
    def generate_introduction(self):
        return self.generate_greeting() + "My name is EVA. I'm the official UPF virtual assistant. "+self.generate_services_list()

    def generate_authors(self):
        return "I was created by three UPF students: Igor, Guillem and Dongjun."
    def generate_bye(self):
        return "Bye! I hope to see you soon :)."
    def generate_youre_welcome(self):
        return "You're welcome!"
    def generate_greeting(self):
        return "Hello! " + self.generate_introduction()
    def generate_services_list(self):
        return "I can help you with any question you have about the university. Let me give you some examples:\n" \
               "* In which room can I find professor xxxx?\n" \
               "* How many faculties does the upf have?\n" \
               "* Does the UPF have a Computer Science course?\n" \
               "* Which masters does the UPF offer?\n" \
               "* At which time does the library close?"
    def generate_joke_or_fun_fact(self, joke_or_fact):
        prompt = f"Generate a {joke_or_fact}"
        if (joke_or_fact == 'funny_joke'):
            return pyjokes.get_joke()
        else:
            return randfacts.get_fact()

    def generate_date(self):
        return "The current date in Barcelona is " + date.today().strftime("%d/%m/%Y")

    def generate_time(self):
        return "The current time in Barcelona is " + datetime.now().strftime("%H:%M")

    def generate_weather(self):

        city = "Barcelona"
        country = "ES"
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country}&appid={self.weather_api_key}"
        weather_data = requests.get(weather_url).json()
        weather = weather_data["weather"][0]["description"]
        temperature = weather_data["main"]["temp"] - 273.15
        temperature = round(temperature, 2)
        return f"The weather in {city} is {weather} and the temperature is {temperature}°C."

    def generate_spelling(self,word):
        output = "The word " + word + " is spelled as: "
        for character in word:
            output +=character + " - "
        return output[:-3]

    def generate_definition(self,word):
        # Get meaning of word "Code"
        mn = self.dc.meaning(word)
        mn = list(mn.values())[0][0]
        return "This is what I found about the word \"" + word + "\": " + mn

    def test_everything_blue(self):
        print("")
        print(self.generate_joke_or_fun_fact("funny joke"))
        print(self.generate_joke_or_fun_fact("fun fact"))
        print(self.generate_date())
        print(self.generate_definition("professor"))
        print(self.generate_spelling("university"))
        print(self.generate_time())
        print(self.generate_weather())
        print("")


textGen = TextGenerator()
textGen.test_everything_blue()
