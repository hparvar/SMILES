# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:24:31 2020

@author: h_par
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

from rdkit import Chem
from utils import mol2alt_sentence, show_results, plot_training_results


def load_data_single(filename):
    df = pd.read_csv(filename)
    target = df['P1']
    
    return df.smiles, target

def map_moleculs(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mol_sentences = [mol2alt_sentence(x) for x in mols]
    vocab = np.unique([x for l in mol_sentences for x in l])  # array of unique substructures (Morgan identifiers)
    num_words = len(vocab)  # number of unique substructures
    # Create a mapping of Morgan identifiers to integers between 1 and num_words
    word_map = dict()
    for i in range(len(vocab)):
        word_map[vocab[i]] = i+1
    # mol_map is like mol_sentences but Morgan identifiers are replaced by their value in wordMap
    mol_map = []
    for m in mol_sentences:
        mol_map.append([word_map[s] for s in m])
    mol_map = np.array(mol_map, dtype=object)
    
    max_seq_len = max([len(mol_map[i]) for i in range(len(mol_map))])  # length of longest molecule
    
    return mol_map, word_map, max_seq_len, vocab, num_words


smiles, target = load_data_single('dataset_single.csv')
mol_map, word_map, max_seq_len, vocab, num_words = map_moleculs(smiles)


X, X_test, y, y_test = train_test_split(mol_map, target, test_size=0.1, random_state=11)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=11)
# Make sequences have same length with 0 padding on both sides
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_seq_len)
X_val = keras.preprocessing.sequence.pad_sequences(X_val, maxlen=max_seq_len)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_seq_len)

y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_val = keras.utils.to_categorical(y_val, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)
  

embedding_length = 32
model_name = 'model_2' 
# ToDo: Check if the model exist


# Designing network
model = keras.Sequential()
model.add(keras.layers.Embedding(num_words+1, embedding_length, input_length=max_seq_len))
model.add(keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling1D(pool_size=3))
model.add(keras.layers.LSTM(100))
model.add(keras.layers.Dense(2, activation='softmax'))

model.summary()



#Compiling the model
model.compile(optimizer=keras.optimizers.Adam(lr=0.00001), loss="categorical_crossentropy", metrics=['accuracy'])

#Training the model
lr_reduce = keras.callbacks.ReduceLROnPlateau(min_lr=0.000001)
mcp_save = keras.callbacks.ModelCheckpoint('model/' + model_name + '.h5', save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_val, y_val), callbacks=[mcp_save, lr_reduce], shuffle=True)

plot_training_results(history, 'model/' + model_name)

# Saving the model.json
model_json = model.to_json()
with open("model/" + model_name + '_model.json', 'w') as json_file:
    json_file.write(model_json)

# Loading the model

# loading json and creating model
json_file = open("model/" + model_name + '_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model/" + model_name + ".h5")
print("Loaded model from disk")


# evaluate loaded model on test data

loaded_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
score = loaded_model.evaluate(X_train, y_train, verbose=0)
print("Training %s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
score = loaded_model.evaluate(X_val, y_val, verbose=0)
print("Validation %s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("Test %s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

#Predictions with test dataset
def predict(model, data):
    preds = model.predict(data, batch_size=1, verbose=1)
    y_pred = preds.argmax(axis=1)
    
    return y_pred

predictions = predict(loaded_model, X_test)


    
show_results(loaded_model, model_name, X_test, y_test, 'Test')
show_results(loaded_model, model_name, X_val, y_val, 'Validation')
show_results(loaded_model, model_name, X_train, y_train, 'Train')


