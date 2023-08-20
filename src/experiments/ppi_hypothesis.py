import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from src.paths import *
from src.datasets.pdb import PDB
from src.utilities import *
from src.embeddings.lda import LDA
from src.experiments.ppi_helpers.ppi_functions import *


# Seed script
np.random.seed(42)

text_list = read_list(os.path.join(LISTS_DIRECTORY, "train.txt"))
text_list, pairs = make_doc_list(text_list)

hypothesis_directory = os.path.join(MODEL_DIRECTORY, "hypothesis_test")
if not os.path.isdir(hypothesis_directory):
    os.mkdir(hypothesis_directory)
topics = 10
lda_model = LDA(doc_list=text_list, num_topics=topics)
lda_model.save_embedding(os.path.join(EMBEDDING_DIRECTORY, "PDB_" + str(topics) + "_lda_hypothesis.embeddding.npy"))
lda_model.save_lda_model(os.path.join(hypothesis_directory, "PDB_" + str(topics) + "_lda_hypothesis.model"))
embeddings = lda_model.embedding

samples = embeddings.shape[0]
distance_array = np.empty([pairs, pairs])
coeffictient_array = np.empty([pairs, pairs])

receptor_embeddings = embeddings[:pairs, :]
ligand_embeddings = embeddings[pairs:, :]


for i in tqdm(range(int(pairs))):
    distance_array[i, :] = bhattacharyya_dist(receptor_embeddings[i, :], ligand_embeddings)

diagonal = np.diagonal(distance_array)
true_mean = np.mean(diagonal)
exclude_diag = remove_diag(distance_array)

proportion_list = []
flattened_distance_array = distance_array.reshape(-1)

proportion_list = Parallel(n_jobs=5)(delayed(sample_and_mean)(flattened_distance_array, pairs) for i in tqdm(range(1000000)))


print(f"Mean of distances {np.mean(distance_array)}")
print(f"True mean: {true_mean}")
s = np.sqrt(((flattened_distance_array.shape[0]-1) * np.std(flattened_distance_array) + (diagonal.shape[0] * np.std(diagonal))) / (diagonal.shape[0] + flattened_distance_array.shape[0] + 2))
print(f"Cohens d: {(np.mean(flattened_distance_array)-true_mean)/s}")

count = [i for i in proportion_list if i < true_mean]

plt.hist(proportion_list)
plt.title("p-value:" + str(len(count)/1000000))
plt.axvline(true_mean, color='r', linewidth=2, label="Mean distance between known interactions")
plt.axvline(np.mean(distance_array), color='black', linewidth=2, label="Mean distance between all interactions")

plt.legend()
plt.savefig(os.path.join(LOG_DIRECTORY, "ppi_histogram.png"), format='png', dpi=120)
plt.show()
plt.close()



