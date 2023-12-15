import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import os
import tensorflow as tf 

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

script_dir = os.path.dirname(os.path.abspath(__file__))

data_file_path = os.path.join(script_dir, 'data.json')

with open(data_file_path, 'r') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
       
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(list(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words])))
classes = sorted(list(set(classes)))

with open('texts.pkl', 'wb') as file:
    pickle.dump(words, file)
with open('labels.pkl', 'wb') as file:
    pickle.dump(classes, file)

training = []
output_empty = [0] * len(classes)

max_words = len(words) 

for doc in documents:
    bag = [1 if lemmatizer.lemmatize(word.lower()) in doc[0] else 0 for word in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1


    bag += [0] * (max_words - len(bag))
    output_row += [0] * (len(classes) - len(output_row))

    training.append([bag, output_row])

random.shuffle(training)

train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('model.h5', hist)
print("Model created")
