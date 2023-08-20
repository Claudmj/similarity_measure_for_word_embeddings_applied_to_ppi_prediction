import os
import torch
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src.paths import LISTS_DIRECTORY, DMASIF_EMBEDDING_DIRECTORY, DISTANCE_MATRICES_DIRECTORY
from src.utilities import read_list, load_npy



list_file_path = os.path.join(LISTS_DIRECTORY, "test.txt")
pdb_list = read_list(list_file_path)

subject_list = [i.split("_")[0] + "_" + i.split("_")[1] for i in pdb_list if len(i.split("_")) == 3]
candidate_list = [i.split("_")[0] + "_" + i.split("_")[2] for i in pdb_list if len(i.split("_")) == 3]

def return_distance_torch(subject, type):
    subject_file_name = os.path.join(DMASIF_EMBEDDING_DIRECTORY, subject + "_embeddings.npy")

    if os.path.isfile(subject_file_name):
        subject_features, subject_labels = load_npy(subject_file_name)
    subject_features = torch.from_numpy(subject_features).to(device, non_blocking=True)

    result = []
    for candidate in candidate_list:
        candidate_file_name = os.path.join(DMASIF_EMBEDDING_DIRECTORY, candidate + "_embeddings.npy")
        if os.path.isfile(candidate_file_name):
            candidate_features, candidate_labels = load_npy(candidate_file_name)
        candidate_features = torch.from_numpy(candidate_features).to(device, non_blocking=True)
        distance_array = torch.matmul(subject_features, candidate_features.T)
        if type == "mean":
            prediction = torch.mean(distance_array)
        elif type == "median":
            prediction = torch.median(distance_array)
        elif type == "max":
            prediction = torch.max(distance_array)
        elif type == "min":
            prediction = torch.min(distance_array)
        prediction = prediction.cpu().data.numpy()
        result.append(prediction)

    return result

device = "cuda"
for type in ["max", "mind", "median", "mean"]:
  scores_array = Parallel(n_jobs=1)(delayed(return_distance_torch)(subject, type) for subject in tqdm(subject_list))
  np.save(os.path.join(DISTANCE_MATRICES_DIRECTORY, f"{type}_scores.npy"), np.array(scores_array))



