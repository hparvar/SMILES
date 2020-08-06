# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:37:48 2020

@author: h_par
"""

import os
import tensorflow as tf
from tensorflow import keras
from .utils import show_results, plot_training_results, load_data, extract_features, split_data, extract_molecule_features

    
def train_model1(args):
    # Set arguments
    file_name = args.data_path
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.num_epochs
    model_name = args.model_name
    targets=['P1'] 
    optimizer = keras.optimizers.Adam(lr=lr)
    loss = "sparse_categorical_crossentropy"
    out_dir = args.output_dir

    smiles, target = load_data(file_name, targets=targets)
    features = extract_features(smiles)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, target.values)
    
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

    # Compiling the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    # Training the model
    lr_reduce = keras.callbacks.ReduceLROnPlateau(min_lr=0.00000001)
    if not os.path.exists(os.path.join(out_dir)):
        os.mkdir(os.path.join(out_dir))
    mcp_save = keras.callbacks.ModelCheckpoint(os.path.join(out_dir, model_name + '.h5'), save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[mcp_save, lr_reduce], shuffle=True)
    
    plot_training_results(history, os.path.join(out_dir, model_name))
    
    # Saving the model.json
    model_json = model.to_json()
    with open(os.path.join(out_dir, model_name + '_model.json'), 'w') as json_file:
        json_file.write(model_json)
    

#Predictions with test dataset
def predict(model, data):
    preds = model.predict(data, batch_size=1, verbose=1)
    y_pred = preds.argmax(axis=1)
    return y_pred

def evaluate_model1(args):
    lr = args.lr
    file_name = args.data_path
    targets=['P1'] 
    out_dir = args.output_dir
    model_name = args.model_name
    optimizer = keras.optimizers.Adam(lr=lr)
    loss = "sparse_categorical_crossentropy"
    smiles, target = load_data(file_name, targets=targets)
    features = extract_features(smiles)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, target.values)

    # loading json and creating model
    if not os.path.exists(os.path.join(out_dir, model_name + '_model.json')):
        print("You need to train the model first. Use train model1 with --data_path to train model1.")
        exit()
    json_file = open(os.path.join(out_dir, model_name + '_model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights(os.path.join(out_dir, model_name + ".h5"))
    print("Loaded model from disk")
    
    # evaluate loaded model on test data
    loaded_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    score = loaded_model.evaluate(X_train, y_train, verbose=0)
    print("Training %s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
    score = loaded_model.evaluate(X_val, y_val, verbose=0)
    print("Validation %s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
    score = loaded_model.evaluate(X_test, y_test, verbose=0)
    print("Test %s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
    
    show_results(loaded_model, model_name, X_test, y_test, 'Test')
    show_results(loaded_model, model_name, X_val, y_val, 'Validation')
    show_results(loaded_model, model_name, X_train, y_train, 'Train')

def predict_model1(args):
    # Loading the model
    # loading json and creating model
    out_dir = args.output_dir
    model_name = args.model_name
    smile = args.smile
    if not os.path.isfile(os.path.join(out_dir, model_name + '_model.json')):
        print("You need to train the model first. Use train model1 --data_path to train model1.")
        exit()
                          
    json_file = open(os.path.join(out_dir, model_name + '_model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights(os.path.join(out_dir, model_name + ".h5"))
    print("Loaded model from disk")
    if smile != '':
        smile_features = extract_molecule_features(smile)
        prediction = predict(loaded_model, smile_features)
        print('Predicted P1 for ',smile,':',str(prediction))
