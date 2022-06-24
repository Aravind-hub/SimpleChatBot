import os
import json
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
import numpy as np
import pickle

path = "c:/Users/Aravindh/vs_workspace/ChatBot"
os.chdir(path)

# Loading Data
with open('intents.json') as file:
    data = json.load(file)
    #print(data)

training_sentences = []
training_labels = []
labels = []
responses = []

# Extracting Data
for intent in data['intents']:
    for message in intent['messages']:
        training_sentences.append(message)
        training_labels.append(intent['tag'])

    responses.append(intent['responses'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
        
num_classes = len(labels)
# print(training_sentences)

# Transforming Data

#Label encorder method is to covert variables into numerics for better use in our model
lbl_encode = LabelEncoder()
lbl_encode.fit(training_labels)
training_labels = lbl_encode.transform(training_labels)
# print(training_labels.size)

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
# word count
word_index = tokenizer.word_index
# sequencial word count in each message
sequences = tokenizer.texts_to_sequences(training_sentences)
# print("==============================================================================")
# print(word_index)
# print("==============================================================================")
# print(sequences)
# converts above sequences list of same size by adding zeros at prefix
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=20)
# print(padded_sequences)

# Define model
model = Sequential()
model.add(Embedding(1000, 16, input_length=20))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

model.summary()

# Train model
history = model.fit(padded_sequences, np.array(training_labels), epochs=500)

# Saving model
model.save("chat_model")

# Saving fitted tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# Saving fitted label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encode, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
