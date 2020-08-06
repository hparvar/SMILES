# Prediction of a drug molecule properties

This repository provides three different deep learning models for drug molecule properties prediction. *Model 1* and *Model 2* predict just one molecule property (P1) and *Model 3* predicts multiple properties of a molecule (P1, P2, ..., P9). 

- **Model 1:** This model gets the SMILE string and turnes it into 2048 bit binary code using the Morgan fingerprint method, then uses a deep learning network (multiple dense layers with Relu and sigmoid activation layers, dropout, batch normalization and softmax layers) to classify the property of the molecule (P1).

- **Model 2:** This model gets the SMILE string directly. Using a vectorization method, I mapped SMILE string into a series of numbers. I used a deep learning network containing the embedding, CNN (1D convolution plus max pooling), LSTM and softmax layers to classify P1.

- **Model 3:** This model is based on multi-task learning and built based on the *Model 1* with similar deep learning architecture by extracting the features from the SMILE string first. There are 9 different properties for each molecule. This model can predict the different properties of a molecule.

## Dataset
There are two different datasets in this project. They both contain the SMILE representation of a molecule (`smiles`) and the molecule ID (`mol_id`) as well as the molecule properties. The first dataset (`dataset_single.csv`) contains just one property (`P1`) as a label or target for classification. The second dataset (`dataset_multi.csv`) contains 9 different properties (`P1, P2, ..., P9`) for multi-task classification as well as `smiles` and `mol_id`. *Model 1* and *Model 2* use the first dataset and *model 3* uses the second dataset. 

For all models, 10% of the data is used for tests (500 samples), and the remaining is used for training (4049 samples) and validation (450 samples). 

I have to mention that the dataset is unbalanced because of that I show the weighted average for the accuracy of the model.

## Results
All the trained models as well as some metrics in csv file, confusion matrix and training loss and accuracy plots will be saved in the `model` directory basen on the `model_name`.

## Model 1
### Pre-processing
In the pre-processing phase, we should convert the SMILE representation into the binary code. The main function for extracting the features is `fingerprint_features` which converts the SMILE string into a bit vector with a fixed length using the Morgan Fingerprint method. There are two functions in the `utils.py` file named `extract_features` and `extract_molecule_features`. These two functions convert the output of the `fingerprint_features` into a numpy array which can be used by the model. `extract_features` is used for preparing data for training, validation and test and `extract_molecule_features` is used for predict a single SMILE.

### Network Architecture
Here is a simple deep learning architecture for our classification task. I used *Adam* optimizer with learning rate *0.00001*, *sparse categorical cross entropy* loss function and batch size *32*.
```python
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
```
I have to mention that this architecture and its hyper-parameters are not optimum and there might be more advanced architectures that could generate better accuracy. For example, we can utilize Neural Architecture Search (NAS) and Hyper Parameter Optimization (HPO) methods to find the best architecture and hyper-parameters. 
## Model 2
### Pre-processing
This model gets the SMILE string directly. We should word embedding method to convert the SMILE string into a series of numbers. For vectorization, we should first calculate Morgan fingerprint and calculate identifiers of substructures as a sentence for all SMILEs. I used `mol2alt_sentence` function from this [link](https://github.com/samoturk/mol2vec/blob/master/mol2vec/features.py). Then we should vectorize the obtained sentences to an array of numbers. The function `map_molecules` returns mapped molecules as well as the world map related to all samples of the dataset. Note I saved the world map and the maximum length of the sentences in a JSON file to map a single molecule for the prediction process using `map_single_molecule` function in the `utils.py` file. Moreover, I zero-padded the embedding to maximum length to have a fixed input to my network.
### Network Architecture
I used the folowwing deep learning architecture for *Model 2*. The first layer is embedding layer. I used one dimentionam convolution layer with kernel size *3*  and *Relu* activation function followed by maxpooling layer to extract features on the embedding. Then I used an LSTM network followed by a softmax layer to classify the SMILE string. I used *Adam* optimizer with learning rate *0.00001*, *categorical crossentropy* loss function and batch size *32*.
```python
model = keras.Sequential()
model.add(keras.layers.Embedding(num_words+1, embedding_length, input_length=max_seq_len))
model.add(keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling1D(pool_size=3))
model.add(keras.layers.LSTM(100))
model.add(keras.layers.Dense(2, activation='softmax'))
```
I have to mention that we can define different modern architecture like using BiLSTM to get better results. 
## Model 3
### Pre-Processing
As this model is based on the *Model 1*, I used the pre-processing method that I used in the *Model 1*. But, I had to built targets for all properties using function `extract_multi_targets`.
### Network Architecture
I added 9 separate softmax layers to the architecture of the *model 1* to perform multi-task classification as follow:
```python
base_model = keras.Sequential()
model_input = keras.layers.Input(shape=(X_train.shape[1], ))
base_model.add(model_input)
base_model.add(keras.layers.Dense(512, activation='relu'))
base_model.add(keras.layers.Dense(256, activation='sigmoid'))
base_model.add(keras.layers.Dense(128, activation='sigmoid'))
base_model.add(keras.layers.Dense(64, activation='sigmoid'))
base_model.add(keras.layers.Dense(32, activation='sigmoid'))
base_model.add(keras.layers.BatchNormalization())
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
```
The optimizer and loss function are the same as the *Model 1*, But I defined one separate loss function for each task (*sparse categorical cross-entropy*).
# Requirements
- **Python 3.6**
- tensorflow 2.1.0
- RDKit
- scikit-learn
- pandas
- NumPy
- matplotlib
- seaborn

# Installation
```sh
$ cd 
$ install
```
# Application

# Docker
This package is very easy to install and deploy in a Docker container.

```sh
cd 
docker build 
```
This will create the servier image and pull in the necessary dependencies. 

Once done, run the Docker image and blah blah blah

## Documentation
Read the documentation on [Read the Docs](http://).

```python
from tensorflow import keras
```
