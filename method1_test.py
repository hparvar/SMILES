# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:37:48 2020

@author: h_par
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os, sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels

import tensorflow as tf
from tensorflow import keras

from feature_extractor import fingerprint_features

def print_confusion_matrix(confusion_matrix, class_names, title, modelname,
                           figsize=(10, 7),
                           fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a seaborn heatmap.
    Saves confusion matrix file to jpg file."""
    df_cm = pd.DataFrame(confusion_matrix, index=class_names,
                         columns=class_names, )
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, ax=ax, fmt="d",
                              cmap=plt.cm.Oranges)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0,
                                 ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,
                                 ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.tight_layout()
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.savefig('model/' + modelname + '_' + title + '_confusion_matrix.png')
    plt.show()

def show_results(model, modelname, data, label, title):
    preds = model.predict(data, batch_size=1, verbose=1)
    y_pred = preds.argmax(axis=1)

    y_actual = np.squeeze(label)

    actualdf = pd.DataFrame({'actualvalues': y_actual})
    preddf = pd.DataFrame({'predictedvalues': y_pred})
    finaldf = actualdf.join(preddf)

    finaldf.groupby('actualvalues').count()
    finaldf.groupby('predictedvalues').count()

    report = pd.DataFrame(
        classification_report(y_actual, y_pred, output_dict=True)).T
    report.to_csv(
        'model/' + modelname + '_' + title + '_classification_report.csv')
    print('\nTest Stats\n', classification_report(y_actual, y_pred))
    print_confusion_matrix(confusion_matrix(y_actual, y_pred),
                           unique_labels(y_actual, y_pred), title, modelname)
    



def ExplicitBitVect_to_NumpyArray(bitvector):
    bitstring = bitvector.ToBitString()
    intmap = map(int, bitstring)
    return np.array(list(intmap))

def load_data_single(filename):
    df = pd.read_csv(filename)
    
    features = []
    for i in range(len(df)):
        extracted_example = fingerprint_features(df['smiles'].iloc[i])
        features.append(ExplicitBitVect_to_NumpyArray(extracted_example))
    df['features']=features
    
    X, X_test, y, y_test = train_test_split(df['features'], df['P1'], test_size=0.1, random_state=11)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=11)
    
    return np.vstack(X_train) , np.vstack(X_val) , np.vstack(X_test) , np.vstack(y_train) , np.vstack(y_val) , np.vstack(y_test)


model_name = 'first_run' 
# ToDo: Check if the model exist

# Load and split data
X_train, X_val, X_test, y_train, y_val, y_test =  load_data_single('dataset_single.csv')


# Designing network
model = keras.Sequential()
model.add(keras.layers.Input(shape=(X_train.shape[1], )))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(128, activation='sigmoid'))
model.add(keras.layers.Dense(64, activation='sigmoid'))
model.add(keras.layers.Dense(32, activation='sigmoid'))
model.add(keras.layers.Dense(16, activation='sigmoid'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(2, activation='softmax'))

model.summary()

#Compiling the model
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss="sparse_categorical_crossentropy", metrics=['accuracy'])

#Training the model
lr_reduce = keras.callbacks.ReduceLROnPlateau(min_lr=0.000001)
mcp_save = keras.callbacks.ModelCheckpoint('model/' + model_name + '.h5', save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(X_train, y_train, batch_size=32, epochs=1000, validation_data=(X_val, y_val), callbacks=[mcp_save, lr_reduce], shuffle=True)


# Plotting the Train Valid Loss Graph

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('model/' + model_name + '_loss.png')
plt.show()


# Plotting the Train Valid Accuracy Graph

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('model/' + model_name + '_accuracy.png')
plt.show()

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


