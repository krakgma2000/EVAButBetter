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


df = pd.read_csv("./data/listado.csv")
# model_name = TTS.list_models()[0]
# tts = TTS(model_name)


def findFromCsv(tracker, slot_name, look_for):
    min_distance = 999
    isFind = True
    slot_value = tracker.get_slot(slot_name).lower()
    # try:
    #     df["isIncluded"] = df[look_for].str.lower().str.find(slot_value) 
    #     res = df.loc[df["isIncluded"] != -1]
    #     print(res)
    #     return res
    # except:
    #     for index, ele in df[look_for].iterrows():
    #         distance = lv.distance(ele, slot_value)
    #         if distance == 0:
    #             return ele
    #         if distance < min_distance:
    #             min_distance = distance
    #             ind = index
    #     return df[ind]
    df["isIncluded"] = df[look_for].str.lower().str.find(slot_value) 
    res = df.loc[df["isIncluded"] != -1]
    # print(res.empty)
    if res.empty:
        isFind = False
        for index, ele in df.iterrows():
            distance = 0
            name_array = ele[look_for].lower().split(" ")
            for name in name_array:
                distance += lv.distance(name, slot_value)
                # if distance == 0:
                #     return ele
            if distance < min_distance:
                min_distance = distance
                # ind = index
                res = ele
        # print(ind)
        # return df.irow(ind)
    # print(res['NOMBRE'])
    return res, isFind

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
        # wav = tts.tts("hello world", speaker=tts.speakers[0], language=tts.languages[0])
        # tts.tts_to_file(text = "hello world", speaker=tts.speakers[0], language=tts.languages[0], file_path = "output.wav")
        return []

class ActionDepartmentInfo(Action):

    def name(self) -> Text:
        return "action_department_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        res = findFromCsv(tracker, "ORG", "GRUPO")
        # print(res)
        dispatcher.utter_message(text=msg)
        msg = "There are  " + str(res.shape[0])  + " professors in " + res["GRUPO"].values[0]
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

        # res = findFromCsv(tracker, "ORG", "GRUPO")
        # # print(res)
        # msg = "There are  " + str(res.shape[0])  + " professors in " + res["GRUPO"].values[0]
        msg = "There are xx departments and xx professors in UPF"
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