# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:37:57 2020

@author: h_par
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split

from feature_extractor import fingerprint_features
from rdkit.Chem import AllChem
from rdkit import Chem
def load_data(filename, targets='P1'):
    df = pd.read_csv(filename)
    return df.smiles, df[targets]

def split_data(features, target, val_size=0.1, test_size=0.1, random_state=42):
    X, X_test, y, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def ExplicitBitVect_to_NumpyArray(bitvector):
    bitstring = bitvector.ToBitString()
    intmap = map(int, bitstring)
    return np.array(list(intmap))

def mol2alt_sentence(mol, radius=2):
    """Calculates ECFP (Morgan fingerprint) and returns only the alternating sentence
    Calculates ECFP (Morgan fingerprint) and returns identifiers of substructures as 'sentence' (string).
    Returns a tuple with 1) a list with sentence for each radius and 2) a sentence with identifiers from all radii
    combined.
    NOTE: Words are ALWAYS reordered according to atom order in the input mol object.
    NOTE: Due to the way how Morgan FPs are generated, number of identifiers at each radius is smaller
    Code from https://github.com/samoturk/mol2vec/blob/master/mol2vec/features.py
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float 
        Fingerprint radius
    
    Returns
    -------
    list
        alternating sentence
    combined
    """
    radii = list(range(radius))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)

def extract_features(smiles):
    features = []
    for i in range(len(smiles)):
        extracted_example = fingerprint_features(smiles.iloc[i])
        features.append(ExplicitBitVect_to_NumpyArray(extracted_example))
    # return features
    return np.vstack(features)

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

def plot_training_results(history, model_name_path):
    # Plotting the Train Valid Loss Graph
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(model_name_path + '_loss.png')
    plt.show()
    
    # Plotting the Train Valid Accuracy Graph
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(model_name_path + '_accuracy.png')
    plt.show()



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
    if preds.ndim == 1:
        y_actual = np.squeeze(label)
    else:
        y_actual = label.argmax(axis=1)

    report = pd.DataFrame(
        classification_report(y_actual, y_pred, output_dict=True)).T
    report.to_csv(
        'model/' + modelname + '_' + title + '_classification_report.csv')
    print('\n' + title + 'Stats\n', classification_report(y_actual, y_pred))
    print_confusion_matrix(confusion_matrix(y_actual, y_pred),
                           unique_labels(y_actual, y_pred), title, modelname)
    


def plot_training_results_multitask(history, model_name_path, targets):
    # Plotting the Train Valid Loss Graph
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(model_name_path + '_loss.png')
    plt.show()
    
    for i in targets:
        # Plotting the Train Valid Loss Graph
        plt.plot(history.history[i + '_loss'])
        plt.plot(history.history['val_' + i + '_loss'])
        plt.title('model loss for ' + i)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(model_name_path + '_' + i + '_loss.png')
        plt.show()

        # Plotting the Train Valid Accuracy Graph
        
        plt.plot(history.history[i + '_accuracy'])
        plt.plot(history.history['val_' + i + '_accuracy'])
        plt.title('model accuracy for ' + i)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(model_name_path + '_' + i + '_accuracy.png')
        plt.show()



def show_results_multitask(model, modelname, data, label, title, targets):
    preds = model.predict(data, batch_size=1, verbose=1)
    # y_pred = preds.argmax(axis=1)
    y_pred = []
    y_actual = []
    for i in range(len(preds)):
        y_pred.append(preds[i].argmax(axis=1))
        y_actual.append(label[i].values)
   
    for i in range(len(targets)):
        report = pd.DataFrame(
            classification_report(y_actual[i], y_pred[i], output_dict=True)).T
        report.to_csv('model/' + modelname + '_' + title + '_' + targets[i] + '_classification_report.csv')
        print('\n' + title + '_' + targets[i] + 'Stats\n', classification_report(y_actual[i], y_pred[i]))
        print_confusion_matrix(confusion_matrix(y_actual[i], y_pred[i]),
                                unique_labels(y_actual[i], y_pred[i]), title + '_' + targets[i], modelname)
