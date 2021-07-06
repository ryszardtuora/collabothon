import distance
import json

from rasa.nlu.components import Component
from rasa.nlu import utils
from rasa.nlu.model import Metadata


class CountryExtractor(Component):
    name = "country_extractor"
    provides = ["entities"]
    requires = []
    defaults = {}
    language_list = ["en"]

    def __init__(self, component_config=None):
        super().__init__(component_config)
        countries_file = component_config["countries_file"]
        with open(countries_file) as f:
            self.countries = json.load(f)

    def train(self, training_data, cfg, **kwargs):
        pass

    def convert_to_rasa(self, value, confidence):
        entity = {"value": value,
                  "confidence": confidence,
                  "entity": "country",
                  "extractor": "country_extractor"
                  }
        return entity

    def weighing(self, dist_value, w1, w2):
        maxlen = max([len(w1), len(w2)])
        weighted = dist_value 
        return weighted 

    def get_country(self, token):
        dists = [(distance.levenshtein(token, k, max_dist=1), k) for k in self.countries]
        pos_dists = [d for d in dists if d[0] >= 0]
        ranking = sorted(pos_dists)
        if ranking:
            dist, top = ranking[0]
            return top
        else:
            return None

    def process(self, message, **kwargs):
        entities = []
        try:
            text = message.data['text']
        except KeyError:
            return None
        tokens = text.split()
        for tok in tokens:
            entity = self.get_country(tok)
            if entity:
                entities.append(self.convert_to_rasa(entity, 1))
        if entities:
            message.set("entities", entities, add_to_output=True)
        with open("ab", "w") as f:
            f.write(str(entities))
 
        
    def persist(self, file_name, model_dir):
        pass


