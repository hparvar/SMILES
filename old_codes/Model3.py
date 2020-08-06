# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:37:48 2020

@author: h_par
"""


from tensorflow import keras

from utils import plot_training_results_multitask, load_data, extract_features, split_data, show_results_multitask

# parameters --------------------------------------------
targets=['P2', 'P1', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'] 
file_name = 'dataset_multi.csv'
lr = 0.00001
optimizer = keras.optimizers.Adam(lr=lr)
loss = "sparse_categorical_crossentropy"
batch_size = 32
epochs = 300
model_name = 'model_3' # ToDo: Check if the model exist
# -------------------------------------------------------

smiles, target = load_data(file_name, targets=targets)
features = extract_features(smiles)
# Split data to train, validation and test
X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, target)

# Add same loss function for all tasks
loss_list = []
for i in range(len(targets)):
    loss_list.append(loss)
loss = loss_list


def extract_multi_targets(y_train, y_val, y_test, targets):
    train = []
    val = []
    test = []
    for i in targets:
        train.append(y_train[i])
        val.append(y_val[i])
        test.append(y_test[i])
    
    return train, val, test
    

y_train, y_val, y_test = extract_multi_targets(y_train, y_val, y_test, targets)





# Designing network


base_model = keras.Sequential()
model_input = keras.layers.Input(shape=(X_train.shape[1], ))
base_model.add(model_input)
base_model.add(keras.layers.Dense(512, activation='relu'))
base_model.add(keras.layers.Dense(256, activation='sigmoid'))
base_model.add(keras.layers.Dense(128, activation='sigmoid'))
base_model.add(keras.layers.Dense(64, activation='sigmoid'))
base_model.add(keras.layers.Dense(32, activation='sigmoid'))
base_model.add(keras.layers.BatchNormalization())

base_model.summary()


y1 = keras.layers.Dense(16, activation='sigmoid')(base_model(model_input))
y1 = keras.layers.Dense(2, activation='softmax', name=targets[0])(y1)

y2 = keras.layers.Dense(16, activation='sigmoid')(base_model(model_input))
y2 = keras.layers.Dense(2, activation='softmax', name=targets[1])(y2)

y3 = keras.layers.Dense(16, activation='sigmoid')(base_model(model_input))
y3 = keras.layers.Dense(2, activation='softmax', name=targets[2])(y3)

y4 = keras.layers.Dense(16, activation='sigmoid')(base_model(model_input))
y4 = keras.layers.Dense(2, activation='softmax', name=targets[3])(y4)

y5 = keras.layers.Dense(16, activation='sigmoid')(base_model(model_input))
y5 = keras.layers.Dense(2, activation='softmax', name=targets[4])(y5)

y6 = keras.layers.Dense(16, activation='sigmoid')(base_model(model_input))
y6 = keras.layers.Dense(2, activation='softmax', name=targets[5])(y6)

y7 = keras.layers.Dense(16, activation='sigmoid')(base_model(model_input))
y7 = keras.layers.Dense(2, activation='softmax', name=targets[6])(y7)

y8 = keras.layers.Dense(16, activation='sigmoid')(base_model(model_input))
y8 = keras.layers.Dense(2, activation='softmax', name=targets[7])(y8)

y9 = keras.layers.Dense(16, activation='sigmoid')(base_model(model_input))
y9 = keras.layers.Dense(2, activation='softmax', name=targets[8])(y9)


model = keras.Model(inputs=model_input, outputs=[y1, y2, y3, y4, y5, y6, y7, y8, y9])

model.summary()

#Compiling the model
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
# model.compile(optimizer=optimizer, loss=loss, metrics=["sparse_categorical_crossentropy"])

#Training the model
lr_reduce = keras.callbacks.ReduceLROnPlateau(min_lr=0.000001)
mcp_save = keras.callbacks.ModelCheckpoint('model/' + model_name + '.h5', save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[mcp_save, lr_reduce], shuffle=True)

plot_training_results_multitask(history, 'model/' + model_name, targets)

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


# evaluate loaded model on train, validat and test data

loaded_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
score = loaded_model.evaluate(X_train, y_train, verbose=0)
for i in range(len(score)):
    print("Training %s: %.2f%%" % (loaded_model.metrics_names[i], score[i] * 100))
score = loaded_model.evaluate(X_val, y_val, verbose=0)
for i in range(len(score)):
    print("Validation %s: %.2f%%" % (loaded_model.metrics_names[i], score[i] * 100))
score = loaded_model.evaluate(X_test, y_test, verbose=0)
for i in range(len(score)):
    print("Test %s: %.2f%%" % (loaded_model.metrics_names[i], score[i] * 100))

#Predictions with test dataset
def predict(model, data):
    preds = model.predict(data, batch_size=1, verbose=1)
    y_pred = []
    for i in range(len(preds)):
        y_pred.append(preds[i].argmax(axis=1))
    
    return y_pred

predictions = predict(loaded_model, X_test)

    
show_results_multitask(loaded_model, model_name, X_test, y_test, 'Test', targets)
show_results_multitask(loaded_model, model_name, X_val, y_val, 'Validation', targets)
show_results_multitask(loaded_model, model_name, X_train, y_train, 'Train', targets)

