"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 April 2023
@Version  : 0.1
@Desc     :
"""

import os
import numpy as np
from tqdm import tqdm
import torch

from src.paths import *


class DmasifEmbeddings:
    """
    Reads a .npy embedding file for the subject and candidate.

    Args:
        protein_code: str
    """
    def __init__(self, subject, candidate, device="cpu"):
        self.device = device
        self.subject_file_name = os.path.join(DMASIF_EMBEDDING_DIRECTORY, subject + "_embeddings.npy")
        self.candidate_file_name = os.path.join(DMASIF_EMBEDDING_DIRECTORY, candidate + "_embeddings.npy")

        if os.path.isfile(self.subject_file_name):
            self.subject_features, self.subject_labels = self.load_npy(self.subject_file_name)
        if os.path.isfile(self.candidate_file_name):
            self.candidate_features, self.candidate_labels = self.load_npy(self.candidate_file_name)


    def get_sigmoid_distances_torch(self):
        self.subject_features = torch.from_numpy(self.subject_features).to(self.device)
        self.candidate_features = torch.from_numpy(self.candidate_features).to(self.device)
        self.distance_array = torch.matmul(self.subject_features, self.candidate_features.T)

        return self.distance_array


    def get_sigmoid_distances(self):
        self.distance_array = np.matmul(self.subject_features, self.candidate_features.T)
        # self.predictions_sigmoid = 1 / (1 + np.exp(-dot_product))
        # self.distance_array = np.empty([self.subject_features.shape[0], self.candidate_features.shape[0]])
        # for i in tqdm(range(self.subject_features.shape[0])):
        #     self.distance_array[i, :] = self.distance(self.subject_features[i, :], self.candidate_features)

        return self.distance_array

    def get_interaction_prediction_torch(self, type="mean"):
        self.distance_array = abs(self.distance_array)
        if type == "mean":
            prediction = torch.mean(self.distance_array)
        elif type == "median":
            prediction = torch.median(self.distance_array)

        return prediction.cpu().numpy()

    def get_interaction_prediction2(self, type="mean"):
        self.distance_array = abs(self.distance_array)
        if type == "mean":
            prediction = np.mean(self.distance_array)
        elif type == "median":
            prediction = np.median(self.distance_array)

        return prediction

    def get_interaction_prediction(self, interaction_threshold=1):
        # Method 1:
        # For each point of target, find if there are any matched points with binder
        # Return the amount of points that had matches above threshold
        # matched_points = np.where(np.any(predictions > prediction_thres)) # for single
        self.distance_array = abs(self.distance_array)
        mean = np.mean(self.distance_array)
        # filtered = min[min < 1].shape[0]
        # matched_points_target = np.where(np.any((self.predictiona_sigmoid > prediction_thres), axis=1))
        # matched_points_binder = np.where(np.any((self.predictiona_sigmoid > prediction_thres), axis=0))
        # print(len(matched_points_target[0]))
        # print(len(matched_points_binder[0]))
        # return len(matched_points_target)
        # if filtered >= 300:
        #     result = 1
        # else:
        #     result = 0

        return mean

    @staticmethod
    def load_npy(file_name):
        embeddings = np.load(file_name)
        features = embeddings[:, :-2]
        labels = embeddings[:, -1]

        return features, labels

    @staticmethod
    def distance(subject_features, candidate_features):
        return np.linalg.norm(subject_features - candidate_features, axis=1)
