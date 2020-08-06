# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:24:31 2020

@author: h_par
"""
import os
import json
import tensorflow as tf
from tensorflow import keras
from .utils import show_results, plot_training_results, load_data, map_molecules, split_data, map_single_molecule


def prepare_data(features,target,max_seq_len):
  
    # Split data to train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, target)
    
    # Make sequences have same length with 0 padding on both sides
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_seq_len)
    X_val = keras.preprocessing.sequence.pad_sequences(X_val, maxlen=max_seq_len)
    X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_seq_len)
    
    # Convert labels to onehot
    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_val = keras.utils.to_categorical(y_val, num_classes=2)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model2(args):
    targets=['P1'] 
    file_name = args.data_path
    lr = args.lr
    optimizer = keras.optimizers.Adam(lr=lr)
    loss = "categorical_crossentropy"
    batch_size = args.batch_size
    epochs = args.num_epochs
    model_name = args.model_name # ToDo: Check if the model exist
    embedding_length = args.embedding_length
    out_dir = args.output_dir
    if not os.path.exists(os.path.join(out_dir)):
        os.mkdir(os.path.join(out_dir))
    smiles, target = load_data(file_name, targets=targets)
    features, word_map, max_seq_len, vocab, num_words = map_molecules(smiles)
    # Save mapping info to jason file
    map_info = {}
    map_info['word_map'] = word_map
    map_info['max_seq_len'] = max_seq_len
    with open(os.path.join(out_dir,'map_info_' + model_name + '.json'), 'w') as f:
        json.dump(map_info, f)
    
    X_train,X_val, X_test, y_train, y_val, y_test = prepare_data(features, target, max_seq_len)

    # Designing network
    model = keras.Sequential()
    model.add(keras.layers.Embedding(num_words+1, embedding_length, input_length=max_seq_len))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=3))
    model.add(keras.layers.LSTM(100))
    model.add(keras.layers.Dense(2, activation='softmax'))
    
    model.summary()

    # Compiling the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    #Training the model
    lr_reduce = keras.callbacks.ReduceLROnPlateau(min_lr=0.000001)
    mcp_save = keras.callbacks.ModelCheckpoint(os.path.join(out_dir, model_name + '.h5'), save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[mcp_save, lr_reduce], shuffle=True)
    
    plot_training_results(history, os.path.join(out_dir,model_name))

    # Saving the model.json
    model_json = model.to_json()
    with open(os.path.join(out_dir, model_name + '_model.json'), 'w') as json_file:
        json_file.write(model_json)


def predict(model, data):
    preds = model.predict(data, batch_size=1, verbose=1)
    y_pred = preds.argmax(axis=1)
    
    return y_pred

def evaluate_model2(args):
    targets=['P1'] 
    file_name = args.data_path
    lr = args.lr
    optimizer = keras.optimizers.Adam(lr=lr)
    loss = "categorical_crossentropy"
    model_name = args.model_name
    out_dir = args.output_dir
    smiles, target = load_data(file_name, targets=targets)
    features, word_map, max_seq_len, vocab, num_words = map_molecules(smiles)
    if not os.path.exists(os.path.join(out_dir, model_name + '_model.json')):
        print("You need to train the model first. Use train model2 with --data_path to train model2.")
        exit()
    # loading json and creating model
    json_file = open(os.path.join(out_dir, model_name + '_model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights(os.path.join(out_dir, model_name + ".h5"))
    print("Loaded model from disk")
    
    X_train,X_val, X_test, y_train, y_val, y_test = prepare_data(features, target, max_seq_len)

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


def predict_model2(args):
    model_name = args.model_name
    smile = args.smile
    out_dir = args.output_dir
    if not os.path.exists(os.path.join(out_dir, model_name + '_model.json')):
        print("You need to train the model first. Use train model2 with --data_path to train model2.")
        exit()
    json_file = open(os.path.join(out_dir, model_name + '_model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    
    if smile != '':
        f = open(os.path.join(out_dir,'map_info_' + model_name + '.json'))
        map_info = json.load(f)
        smile_features = keras.preprocessing.sequence.pad_sequences(map_single_molecule(smile, map_info['word_map']), maxlen= map_info['max_seq_len'])
       
        prediction = predict(loaded_model, smile_features)
        print('Predicted P1 for ',smile,':',str(prediction))
