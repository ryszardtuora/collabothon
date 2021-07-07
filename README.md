# collabothon

This repository contains the code for Covidius - a pandemic chatbot using RASA.

### The files have the following functions:
#### Base files:
  - config.yml : the NLU and DM pipeline specification, listing all the components and their parameters 
  - endpoints.yml : connection with external applications (servers, messaging applications) 
  - credenrtials.yml : logging information for connecting with external services
  - domain.yml : listing of all the intents, entities, actions, forms, and responses for the chatbot 
  - data/nlu.yml : training data for the NLU component (intents + entities) 
  - data/rules.yml : deterministic rules for DM to choose actions
  - data/stories.yml : examples of longer interactions for training DM 

#### Extensions:
  - CountryExtractor : NLU component for recognizing countries in user messages based on Levenshtein distance and a list of country names
  - countries.json : a list of country names for CountryExtractor
  - actions : Custom actions for QA, and fetching the data from APIs
  - scrape.py : a script for scraping QA data from CDC, and precalculating the vector representations and saving them
  - QA.json : question:answer pairings of QA data 
  - emb.npy : numpy matrix representing the QA questions 


