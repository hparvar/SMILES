# Prediction of a drug molecule properties

This repository provides three different deep learning models for drug molecule properties prediction. *Model 1* and *Model 2* predict just one molecule property (P1) and *Model 3* predicts multiple properties of a molecule (P1, P2, ..., P9).

- **Model 1:** This model gets the SMILE string and turnes it into 2048 bit binary code using the Morgan fingerprint method, then uses a deep learning network (multiple dense layers with Relu and sigmoid activation layers, dropout, batch normalization and softmax layers) to classify the property of the molecule (P1).

- **Model 2:** This model gets the SMILE string directly. Using a vectorization method, I mapped SMILE string into a series of numbers. I used a deep learning network containing the embedding, CNN (1D convolution plus max pooling), LSTM and softmax layers to classify P1.

- **Model 3:** This model is based on multi-task learning and built based on the *Model 1* with similar deep learning architecture by extracting the features from the SMILE string first. There are 9 different properties for each molecule. This model can predict the different properties of a molecule.

## Dataset
In this project, two different datasets were used. They both contain the SMILE representation of a molecule (`smiles`) and the molecule ID (`mol_id`) as well as the molecule properties. The first dataset (`dataset_single.csv`) contains just one property (`P1`) as a label or target for classification. *Model 1* and *Model 2* use the first dataset.
The second dataset (`dataset_multi.csv`) contains 9 different properties (`P1, P2, ..., P9`) for multi-task classification as well as `smiles` and `mol_id`. This dataset is used to train, evaluate, and predict by *model 3*.

## Data preparation
For all models, 10% of the data is used for tests (500 samples), and the remaining is used for training (4049 samples) and validation (450 samples).
To design train, validation and test sets I used 'sklearn.model_selection'.

Both given datasets are unbalanced. Therefore the weighted average is shown for the accuracy of each model. This is the simplest way to tackle unbalanced data, however, it is not the most efficient method. A more efficient approach is balancing the dataset with oversampling (if collecting more data is not possible or cheap).  

## Results
All the trained models as well as some metrics in CSV file, confusion matrix and training loss and accuracy plots will be saved in the `model` directory based on the `model_name`.

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
This architecture and its hyper-parameters are not optimum and to obtain better accuracy we need to fine-tune these hyperparameters.
Moreover, more advanced architectures can be found using Neural Architecture Search (NAS) and Hyper Parameter Optimization (HPO) methods. Especially various NAS methods have been introduced recently that can find optimal architectures for a given dataset using reinforcement learning methods.

## Model 2
### Pre-processing
This model gets the SMILE string directly. We should word embedding method to convert the SMILE string into a series of numbers. For vectorization, we should first calculate Morgan fingerprint and calculate identifiers of substructures as a sentence for all SMILEs. I used `mol2alt_sentence` function from this [link](https://github.com/samoturk/mol2vec/blob/master/mol2vec/features.py). Then we should vectorize the obtained sentences to an array of numbers. The function `map_molecules` returns mapped molecules as well as the world map related to all samples of the dataset. Note I saved the world map and the maximum length of the sentences in a JSON file to map a single molecule for the prediction process using `map_single_molecule` function in the `utils.py` file. Moreover, I zero-padded the embedding to maximum length to have a fixed input to my network.
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
Recurrent neural networks (RNNs) are very good at detecting the sequence of events. I believe we can define different modern architecture using LSTMs, BiLSTM, or GRUs (from RNN family) to get better results.

## Model 3
### Pre-Processing
This model is basically the *Model 1*. I used the pre-processing method that I used in the *Model 1*. The main difference between *Model 1* and *Model 3* is target properties. To design targets for *Model 3* a function named `extract_multi_targets` is used.
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
The optimizer and loss function are the same as the *Model 1*. Since this is a multi-task learning problem, I defined one separate loss function for each task (*sparse categorical cross-entropy*).

# Deliverables
I divided deliverables into two folders named:
- HP_servier_package
This folder includes `Dockerfile` and `setup.py`. It also includes `Servier_package` folder that contains all scripts for the prediction of a drug molecule properties.
- HP_api
This folder includes flask api scripts.

# Installation without docker
Prepare your servier environment with these packages:
```sh
$ wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda
$ export PATH=~/miniconda/bin:$PATH
$ conda update -n base conda
$ conda create -y --name servier python=3.6
$ conda activate servier
$ conda install -y tensorflow=2.1.0
$ conda install -y scikit-learn
$ conda install -y -c conda-forge rdkit
$ conda install -y -c conda-forge matplotlib
$ conda install -y seaborn
$ conda install -y -c conda-forge flask-restful
```
To install the package after downloading the `HP_servier_package`:
```sh
$ cd /path/to/HP_servier_Package
$ python setup.py develop (or install)
```
After installing the package you can run the models with proper arguments.
**Example 1:** to train `model1` you can use this command:
```sh
$ servier train model1 --data_path ../path/to/dataset_single.csv
```
Since dataset is not included in this repository, you need to specify the path to the proper dataset when calling train or evaluate tasks.
**Example 2:** to evaluate model3 you can use this command:
```sh
$ servier evaluate model3 --data_path ../path/to/dataset_multi.csv
```
**Example 3:** to predict model2 you can use this command:
```sh
$ servier predict model2 --smile Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C
```
You can also change other arguments including `lr`,`batch_size`, `num_epochs`, etc.

# Api
To use provided api please download `HP_api` and follow these steps:
```sh
$ cd /path/to/HP_api
$ mkdir model
$ cd model
$ cp /path/to/model1_model.json .
$ cp /path/to/model1.h5 .
$ python ../HP_servier_api.py
```
Now you can open a browser with the given address and use `/predict/<SMILE>` route to predict `P1` property of a SMILE string using a trained `model1`.
Example: `http://127.0.0.1:5000/predict/Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1`

# Docker
All dependencies and environment properties can be set using the Dockerfile available in the `HP_servier_package` folder.
To run the Dockerfile please go to the directory of `Dockerfile` and run:

```sh
$ sudo docker build -t servier .
```
When it is successfully installed you can run the built image:
```sh
$ sudo docker run servier
```