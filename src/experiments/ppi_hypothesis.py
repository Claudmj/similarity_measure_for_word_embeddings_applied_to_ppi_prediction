import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from src.paths import *
from src.utilities import *
from src.embeddings.lda import LDA
from src.experiments.ppi_helpers.ppi_functions import *


# Seed script
np.random.seed(42)

n = 10000

text_list = read_list(os.path.join(LISTS_DIRECTORY, "pdb_list_5481.txt"))
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


######################
# Bootstrap
######################

diagonal = np.diagonal(distance_array) # get known interaction distances
true_mean = np.mean(diagonal) # get mean of known interaction distances
exclude_diag = remove_diag(distance_array) # get non interacting distances
exclude_diag = exclude_diag.reshape(-1)
test_statistic = np.mean(exclude_diag) # get mean distance between all the embeddings that are not part of a known PPI

# Perform simulation under conditions in which we know the null hypothesis is true so shift it over so that the mean really = true_mean
exclude_diag = exclude_diag - test_statistic + true_mean

# Bootstrap 10000 samples and return the mean of each sample. Mean list = 10000 bootstrapped means
sample_size = exclude_diag.shape[0]
mean_list = Parallel(n_jobs=4)(delayed(sample_and_mean)(exclude_diag, sample_size) for i in tqdm(range(n)))

print(f"Mean of distances {test_statistic}")
print(f"True mean: {true_mean}")
s = np.sqrt(((exclude_diag.shape[0]-1) * np.std(exclude_diag) + (diagonal.shape[0] * np.std(diagonal))) / (diagonal.shape[0] + exclude_diag.shape[0] + 2))
print(f"Cohens d: {(test_statistic-true_mean)/s}")

# Calculate p-value as the average number of sample means that are less than the mean distance
# between all the embeddings that are not part of a known PPI
count = [i for i in mean_list if i >= test_statistic]
print("p-value:" + str(len(count)/n))

plt.figure(figsize=(7, 7))
plt.hist(mean_list, range=[0.831, 0.838])
plt.savefig(os.path.join(LOG_DIRECTORY, "ppi_histogram_means.png"), format='png', dpi=120)
plt.show()
plt.close()


plt.figure(figsize=(7, 7))
plt.hist(mean_list, range=[0.6, 2])
plt.axvline(true_mean, color='r', linewidth=2, label="Mean distance between \nembeddigns of known PPIs")
plt.axvline(test_statistic, color='black', linewidth=2, label="Mean distance between \nembeddings that are not \npart of known PPIs")
plt.legend()
plt.savefig(os.path.join(LOG_DIRECTORY, "ppi_histogram_comparison.png"), format='png', dpi=120)
plt.show()
plt.close()



