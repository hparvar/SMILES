# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:13:54 2020

@author: Farzaneh
"""
import os
from flask import Flask
import tensorflow as tf
from tensorflow import keras
from flask_restful import abort, Api, Resource
from flask_restplus import reqparse

from utils import show_results, plot_training_results, load_data, extract_features, split_data, extract_molecul_features

app = Flask(__name__)

def predict(model, data):
    preds = model.predict(data, batch_size=1, verbose=1)
    y_pred = preds.argmax(axis=1)
    return y_pred

# @app.route('/predict',defaults={'smile':'Please enter any smile to get the prediction'})

@app.route('/predict/<smile>')
def projects(smile):
    
    out_dir = 'model' # please enter adrees to the model folder that contains model1_model.json'
    model_name = 'model1'
    
    json_file = open(os.path.join(out_dir, model_name + '_model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights(os.path.join(out_dir, model_name + ".h5"))
    print("Loaded model from disk")
    if smile != '':
        smile_features = extract_molecul_features(smile)
        prediction = predict(loaded_model, smile_features)
        print('Predicted P1 for ',smile,':',str(prediction))

    return "Predicted P1 for this smile is: %d "%prediction

if __name__ == '__main__':
    app.run(debug=True)