import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

# lemmatizáló objektum, ami a szavakat alapformájukra alakítja (pl. "running" -> "run")
lemmatizer = WordNetLemmatizer()

# szándékok és minták betöltése
intents = json.loads(open('intents.json').read())

# pickle fájlok betöltése, amelyek a chatbot tréningje során használt szavakat és osztályokat tartalmazzák
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# mentett, betanított modell betöltése ami az üzenetek osztályozására szolgál
model = load_model('chatbot_model.model')


#mondat takarítása
def clean_up_sentence(sentence):
    # A bemeneti mondatot szavakra bontjuk
    sentence_words = nltk.word_tokenize(sentence)
    # Minden egyes szót lemmatizálunk (pl. többesszámot egyes számmá, igéket alapformává alakít)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# A "bag of words" (szavak zsákja) reprezentáció létrehozása a mondat alapján
def bag_of_words(sentence):
    # a mondatot megtisztítjuk, lemmatizáljuk és szavakra bontjuk
    sentence_words = clean_up_sentence(sentence)
    # üres zsákot hozunk létre a tanító szavak hosszával megegyező méretben (nullákkal feltöltve)
    bag = [0] * len(words)
    # minden egyes szóra ellenőrizzük, hogy szerepel-e a tanító szavak között
    for trained_word in sentence_words:
        for i, word in enumerate(words):
            if word == trained_word: # ha a szó egyezik a tanító szavak egyikével
                bag[i] = 1 # akkor beállítjuk a pozícióját 1-re, jelezve, hogy ez a szó szerepel a mondatban
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]))[0] #output megítélése
    ERROR_THRESHOLD = 0.25 #25% bizonytalanság esetében ne számolja
    results = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD] #enumeráljuk az eredményeket hogy megkapjuk a classes indexét valamint az esélyt

    results.sort(key=lambda x: x[1], reverse = True) #csökkenő sorrendbe való rendezés a valség alapján
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list