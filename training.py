import random
import json # minták és válaszok
import pickle # szavak, cimkék és listák mentése
import numpy as np

import nltk # neural language toolkit
from nltk.stem import WordNetLemmatizer
# Egy szó alapformájára alakításáért felelős (pl. igeidőket és többesszámot normalizál)

# Sequential() -> neurális hálózat felépítése
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# intents.json mintáinak feldolgozása
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) # A mondatokat szavakra bontja, hogy külön értelmezhetők legyenek
        words.extend(word_list) # A szavakat hozzáadja a teljes szavak listájához (extend helyett append-tel külön listákat hozna létre)
        documents.append((word_list, intent['tag'])) # Az adott szavakat az intent címkéjéhez rendeli

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents) # training data

# Minden szó alapformájára alakítása, majd duplikátumok eltávolítása
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words)) #(halmazzá alakítás)

# A címkék halmazának rendezése (arra az esetre, ha lenne duplikátum)
classes = sorted(set(classes))

# A szavak és a címkék mentése pickle fájlba, hogy újra felhasználhatók legyenek
pickle.dump(words, open('words.pkl', 'wb')) # wb -> writing binaries
pickle.dump(classes, open('classes.pkl', 'wb'))

print(words) # lemmatizált szavak

# gépi tanulás előkészítése - numerizálni kell az adatokat, hogy a modell értelmezni tudja

training = []
output_empty = [0] * len(classes) # Üres lista az egyes címkékhez, kezdetben minden értéke 0

# Minden egyes dokumentum feldolgozása a tanító adathoz
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # "Bag of words" létrehozása - ha a szó szerepel a mondatban, értéke 1 lesz
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Kimeneti sor létrehozása, ahol a megfelelő címkéhez tartozó hely értéke 1 lesz
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row]) # Hozzáadjuk az eredményhez a zsákot és a kimeneti sort

# Adatok véletlenszerű keverése
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0]) # bemenetek (szavak zsákja) az első dimenzióban
train_y = list(training[:, 1]) # címkék a második dimenzióban

# Neurális háló létrehozása
model = Sequential()
# bemeneti réteg
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) # 128 neuron
model.add(Dropout(0.5)) # 50% dropout a túlillesztés elkerülésére
model.add(Dense(64, activation='relu')) # rejtett réteg 64 neuronnal
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax')) # kimeneti réteg, amely a valószínűségi eloszlást adja

# Optimalizáló konfigurálása
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Modell összeállítása, kategóriás "kereszthentrópiás veszteségfüggvénnyel" és pontosság metrikával
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Modell tanítása a bemeneti és kimeneti adatokkal
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist) # A modell mentése, hogy később újra lehessen használni

print("Done")
