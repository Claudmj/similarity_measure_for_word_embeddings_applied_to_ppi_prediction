"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 April 2022
@Version  : 0.1
@Desc     : General useful functions
"""
import numpy as np


def read_list(file_name: str):
    with open(file_name, "r") as file:
        lines = file.read().split("\n")
        return lines

def save_list(file_name: str, list: list):
    with open(file_name, "w") as file:
        file.write("\n".join(list))


def remove_diag(array):
    flattened = array.reshape(-1)
    exclude_diag = np.delete(flattened, range(0, flattened.shape[0], array.shape[0] + 1), 0)
    exclude_diag = exclude_diag.reshape(array.shape[0], array.shape[0] - 1)
    return exclude_diag

def load_npy(file_name):
    embeddings = np.load(file_name)
    features = embeddings[:, :-2]
    labels = embeddings[:, -1]

    return features, labels