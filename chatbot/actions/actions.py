# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet 

import json


# API
import requests

api_address = "https://corona.lmao.ninja/v3/covid-19/countries/{}/?yesterday=true&strict=true&query"

class ActionAPI(Action):
  def name(self):
    return "action_api"
    
  def run(self, dispatcher, tracker, domain):
    country = tracker.get_slot("country")
    if not country:
        country = "Poland"
    formatted_address = api_address.format(requests.utils.quote(country))
    response = requests.get(formatted_address)
    data = json.loads(response.content)
    try:
        message = f"Yesterday, in {country} {data['todayCases']} people got sick, and {data['todayDeaths']}\
 people died because of the coronavirus pandemic."
        dispatcher.utter_message(text=message)
    except KeyError:
        dispatcher.utter_message(text="I apologize, but I couldn't find data on this country, maybe you've made a typo?")
    return []


# QA
import numpy
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
emb = numpy.load("emb.npy")
questions, answers = [], []
with open("QA.json") as f:
  qa_data = json.load(f)

for q,a in qa_data:
  questions.append(q)
  answers.append(a)

class ActionQA(Action):
  def name(self):
    return "action_qa"
    
  def run(self, dispatcher, tracker, domain):
    question = tracker.latest_message["text"]
    ques_vec = model.encode(question)
    distances = [cosine(ques_vec, r) for r in emb]
    top = sorted([d for d in distances if d < 0.4])[:3]
    if top:
      indices = [distances.index(d) for d in top]
      chosen = [(questions[i], answers[i], i) for i in indices]
      if len(top) == 1:
        answer = chosen[0][1]
        dispatcher.utter_message(answer) 
        return []
      else:
        message = f"Please select the question which best matches your inquiry."
        buttons = []
        for question, answer, index in chosen:
          payload = '/inform{"qa_answer":"' + str(index) + '"}'
          buttons.append(
              {"title": "{}".format(question),
               "payload": payload})
        dispatcher.utter_button_message(message, buttons)
        return [] 

    else:
      dispatcher.utter_message("I couldn't understand your question. Can you rephrase it?") 
      return []


class ActionQAAnswer(Action):
    def name(self):
        return "action_qa_answer"

    def run(self, dispatcher, tracker, domain):
        answer_index = int(tracker.get_slot("qa_answer"))
        answer = answers[answer_index]
        dispatcher.utter_message(answer)
        return [SlotSet("qa_answer", None)]


