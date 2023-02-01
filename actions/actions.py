# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd;
import os

df = pd.read_csv("./data/listado.csv")
df.set_index(df["NOMBRE"].str.lower(),inplace=True)
# from TTS.api import TTS

# model_name = TTS.list_models()[0]

# tts = TTS(model_name)

# class ActionHelloWorld(Action):

#     def name(self) -> Text:
#         return "action_hello_world"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

#         dispatcher.utter_message(text="Hello World!")

#         wav = tts.tts("hello world", speaker=tts.speakers[0], language=tts.languages[0])
#         tts.tts_to_file(text = "hello world", speaker=tts.speakers[0], language=tts.languages[0], file_path = "output.wav")
#         return []

class ActionDepartInfo(Action):

    def name(self) -> Text:
        return "action_depart_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        professor_name = tracker.get_slot("PERSON").lower()
        msg = "professor " + professor_name  + " is in " + df.loc[professor_name, "GRUPO"] + " and " + professor_name + "'s office room number is " + df.loc[professor_name, "DESPACHO"]
        dispatcher.utter_message(text=msg)
        print(tracker.slots)
        # wav = tts.tts("hello world", speaker=tts.speakers[0], language=tts.languages[0])
        # tts.tts_to_file(text = "hello world", speaker=tts.speakers[0], language=tts.languages[0], file_path = "output.wav")
        return []