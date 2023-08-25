import os
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import random

from src.datasets.pdb import PDB
from src.utilities import read_list
from src.paths import MODEL_DIRECTORY




def bhattacharyya_coeff(arr1, arr2):
    return np.sum(np.sqrt(arr1 * arr2), axis=1)


def bhattacharyya_dist(arr1, arr2):
    coeff = bhattacharyya_coeff(arr1, arr2)
    coeff[coeff == 0] = 0.000001
    return -np.log(coeff)

def cosine_similarity(arr1, arr2):
    return metrics.pairwise.cosine_similarity(arr1, arr2)

def sample_and_mean(flattened_distance_array, pairs):
    distance_sample = np.random.choice(flattened_distance_array, pairs, replace=True)

    return np.mean(distance_sample)

def sample_and_mean_ttest(flattened_distance_array, pairs, true_mean):
    distance_sample = np.random.choice(flattened_distance_array, pairs, replace=True)

    return (np.mean(distance_sample)- 33.02) / (np.std(distance_sample) / np.sqrt(pairs))

def make_doc_list(pdb_list):
    receptor_list = [i.split("_")[0] + "_" + i.split("_")[1] for i in pdb_list if len(i.split("_")) == 3]
    ligand_list = [i.split("_")[0] + "_" + i.split("_")[2] for i in pdb_list if len(i.split("_")) == 3]

    doc_list = []
    for pdb_code in receptor_list:
        pdb = PDB(pdb_code)
        doc_list.append(pdb.pdb_text)

    for pdb_code in ligand_list:
        pdb = PDB(pdb_code)
        doc_list.append(pdb.pdb_text)

    return doc_list, len(receptor_list)


def make_predictions(idx, distances, diagonal, dummy_list):
    prob_list = []
    dummy_prob = []
    for i in tqdm(idx):
        filtered = distances[i, :][distances[i, :] > diagonal[i]]
        prob_list.append(filtered.shape[0]/distances[i, :].shape[0])
        j = int(dummy_list[i])
        dummy_filter = distances[i, :][distances[i, :] > distances[i, j]]
        if diagonal[i] > distances[i, j]:
            count = dummy_filter.shape[0] + 1
        else:
            count = dummy_filter.shape[0]
        dummy_prob.append(count / distances[i, :].shape[0])

    probabilities = prob_list + dummy_prob
    labels = [1]*len(prob_list) + [0] * len(dummy_prob)

    predictions = [1 if prediction > 0.5 else 0 for prediction in probabilities]

    metrics_dict = {}
    metrics_dict["AUC"] = metrics.roc_auc_score(labels, probabilities)
    metrics_dict["Accuracy"] = metrics.accuracy_score(labels, predictions)
    metrics_dict["Precision"] = metrics.precision_score(labels, predictions)
    metrics_dict["Recall"] = metrics.recall_score(labels, predictions)

    print(f"AUC: {metrics_dict['AUC']}")
    print(f"Accuracy: {metrics_dict['Accuracy']}")
    print(f"Precision: {metrics_dict['Precision']}")
    print(f"Recall: {metrics_dict['Recall']}")

    return probabilities, predictions, labels, metrics_dict


def make_predictions_cosine(idx, distances, diagonal, dummy_list):
    prob_list = []
    dummy_prob = []
    for i in tqdm(idx):
        filtered = distances[i, :][distances[i, :] < diagonal[i]]
        prob_list.append(filtered.shape[0]/distances[i, :].shape[0])
        j = int(dummy_list[i])
        dummy_filter = distances[i, :][distances[i, :] < distances[i, j]]
        if diagonal[i] < distances[i, j]:
            count = dummy_filter.shape[0] + 1
        else:
            count = dummy_filter.shape[0]
        dummy_prob.append(count / distances[i, :].shape[0])

    probabilities = prob_list + dummy_prob
    labels = [1]*len(prob_list) + [0] * len(dummy_prob)

    predictions = [1 if prediction > 0.5 else 0 for prediction in probabilities]

    metrics_dict = {}
    metrics_dict["AUC"] = metrics.roc_auc_score(labels, probabilities)
    metrics_dict["Accuracy"] = metrics.accuracy_score(labels, predictions)
    metrics_dict["Precision"] = metrics.precision_score(labels, predictions)
    metrics_dict["Recall"] = metrics.recall_score(labels, predictions)

    print(f"AUC: {metrics_dict['AUC']}")
    print(f"Accuracy: {metrics_dict['Accuracy']}")
    print(f"Precision: {metrics_dict['Precision']}")
    print(f"Recall: {metrics_dict['Recall']}")

    return probabilities, predictions, labels, metrics_dict


def log_kfold_metrics(experiment_name, n_splits, auc, accuracy, precision, recall):
    log_file_name = os.path.join(MODEL_DIRECTORY, experiment_name, experiment_name + ".log")
    log_file = open(log_file_name, "w", encoding="utf-8")

    log_file.write(
        f"{n_splits}fold cross validation:\n"
        f"AUC: {auc} \t"
        f"Accuracy: {accuracy} \t"
        f"Precision: {precision} \t"
        f"Recall: {recall} \n")
    log_file.flush()