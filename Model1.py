# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:37:48 2020

@author: h_par
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow import keras

from feature_extractor import fingerprint_features
from utils import ExplicitBitVect_to_NumpyArray, show_results, plot_training_results


def load_data_single(filename):
    # ToDo: Modify the function for both methods 
    df = pd.read_csv(filename)
    
    features = []
    for i in range(len(df)):
        extracted_example = fingerprint_features(df['smiles'].iloc[i])
        features.append(ExplicitBitVect_to_NumpyArray(extracted_example))
    df['features']=features
    
    X, X_test, y, y_test = train_test_split(df['features'], df['P1'], test_size=0.1, random_state=11)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=11)
    
    return np.vstack(X_train) , np.vstack(X_val) , np.vstack(X_test) , np.vstack(y_train) , np.vstack(y_val) , np.vstack(y_test)


model_name = 'first_run_10k_dropout' 
# ToDo: Check if the model exist

# Load and split data
X_train, X_val, X_test, y_train, y_val, y_test =  load_data_single('dataset_single.csv')


# Designing network
model = keras.Sequential()
model.add(keras.layers.Input(shape=(X_train.shape[1], )))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(128, activation='sigmoid'))
model.add(keras.layers.Dense(64, activation='sigmoid'))
model.add(keras.layers.Dense(32, activation='sigmoid'))
model.add(keras.layers.Dense(16, activation='sigmoid'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(2, activation='softmax'))

model.summary()

#Compiling the model
model.compile(optimizer=keras.optimizers.Adam(lr=0.000001), loss="sparse_categorical_crossentropy", metrics=['accuracy'])

#Training the model
lr_reduce = keras.callbacks.ReduceLROnPlateau(min_lr=0.00000001)
mcp_save = keras.callbacks.ModelCheckpoint('model/' + model_name + '.h5', save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(X_train, y_train, batch_size=32, epochs=10000, validation_data=(X_val, y_val), callbacks=[mcp_save, lr_reduce], shuffle=True)

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

loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
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


