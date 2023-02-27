# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import AllSlotsReset, UserUttered, SlotSet, ActionExecuted, EventType, FollowupAction, Restarted, SessionStarted, SlotSet

import pandas as pd;
import Levenshtein as lv;
# from TTS.api import TTS
from google.cloud import texttospeech
from playsound import playsound


df = pd.read_csv("./data/listado.csv")
# model_name = TTS.list_models()[0]
# tts = TTS(model_name)


def findFromCsv(tracker, slot_name, look_for):
    min_distance = 999
    isFind = True
    slot_value = tracker.get_slot(slot_name).lower()
    df["isIncluded"] = df[look_for].str.lower().str.find(slot_value) 
    res = df.loc[df["isIncluded"] != -1]
    if res.empty:
        isFind = False
        for index, ele in df.iterrows():
            distance = 0
            name_array = ele[look_for].lower().split(" ")
            for name in name_array:
                distance += lv.distance(name, slot_value)
            if distance < min_distance:
                min_distance = distance
                res = ele
    return res, isFind

def textTospeech(text):
    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open("output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')

    playsound("output.mp3")

class ActionProfessorInfo(Action):

    def name(self) -> Text:
        return "action_professor_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        res, isFind = findFromCsv(tracker, "PERSON", "NOMBRE")
        if isFind:
            msg = "Professor " + res["NOMBRE"].values[0]  + " is in " + res["GRUPO"].values[0] + " and the office number is " + res["DESPACHO"].values[0]
        else:
            msg = "Can't find the professor given the name, the professor who has the most similar name is " + res["NOMBRE"] + " in " + res["GRUPO"] + " and the office room is "  + res["DESPACHO"]
        print(tracker.slots)
        dispatcher.utter_message(text=msg)
        textTospeech(msg)
        # wav = tts.tts("hello world", speaker=tts.speakers[0], language=tts.languages[0])
        # tts.tts_to_file(text = "hello world", speaker=tts.speakers[0], language=tts.languages[0], file_path = "output.wav")
        
        return []

class ActionDepartmentInfo(Action):

    def name(self) -> Text:
        return "action_department_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        res, isFind = findFromCsv(tracker, "ORG", "GRUPO")
        # print(res)
        if isFind:
            msg = "There are  " + str(res.shape[0])  + " professors in " + res["GRUPO"].values[0]
        else:
            msg = "Can't find the grupo given the name, the most similar grupo is " + res["GRUPO"] + " there are " + str(res.shape[0]) + " professors in the grupo" 
        print(tracker.slots)
        dispatcher.utter_message(text=msg)
        # print(tracker.slots)
        # wav = tts.tts("hello world", speaker=tts.speakers[0], language=tts.languages[0])
        # tts.tts_to_file(text = "hello world", speaker=tts.speakers[0], language=tts.languages[0], file_path = "output.wav")
        return []
        
class ActionFacultyInfo(Action):

    def name(self) -> Text:
        return "action_faculty_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        msg = "There are 34 departments and 230 professors in UPF, for more information, please provide more details about the question."
        dispatcher.utter_message(text=msg)
        # print(tracker.slots)
        # wav = tts.tts("hello world", speaker=tts.speakers[0], language=tts.languages[0])
        # tts.tts_to_file(text = "hello world", speaker=tts.speakers[0], language=tts.languages[0], file_path = "output.wav")
        return []
        
class SlotsReset(Action):
	def name(self):
		return 'action_slot_reset'
	def run(self, dispatcher, tracker, domain):
			return[AllSlotsReset()]