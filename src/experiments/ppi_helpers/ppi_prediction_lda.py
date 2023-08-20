from src.embeddings.lda import LDA
from src.paths import *
from src.utilities import *
from src.experiments.ppi_helpers.ppi_functions import *

MAKE_DUMMY_LIST = False

# Seed script
np.random.seed(42)

def make_ppi_predictions_lda():
    dummy_list = read_list(os.path.join(LISTS_DIRECTORY, "dummy_id.txt"))
    topics = 10

    model_name = "ppi_pred"
    model_file_path = os.path.join(MODEL_DIRECTORY, model_name + "_" + str(topics) + "_lda",
                                   model_name + "_" + str(topics) + "_lda.model")
    if not os.path.isfile(model_file_path):
        train_list = read_list(os.path.join(LISTS_DIRECTORY, "train.txt"))
        train_list, pairs = make_doc_list(train_list)
        LDA.train(train_list, model_name, dimension=topics)

    text_list = read_list(os.path.join(LISTS_DIRECTORY, "test.txt"))
    text_list, pairs = make_doc_list(text_list)
    lda_model = LDA(doc_list=text_list, num_topics=topics, model_path=model_file_path)
    lda_model.save_embedding(os.path.join(EMBEDDING_DIRECTORY, "PDB_" + model_name + "_" + str(topics) + "_lda.embeddding.npy"))
    embeddings = lda_model.embedding

    distance_array = np.empty([pairs, pairs])
    receptor_embeddings = embeddings[:pairs, :]
    ligand_embeddings = embeddings[pairs:, :]
    for i in tqdm(range(int(pairs))):
        distance_array[i, :] = bhattacharyya_dist(receptor_embeddings[i, :], ligand_embeddings)

    diagonal = np.diagonal(distance_array)
    test_distances = remove_diag(distance_array)
    idx = [i for i in range(int(pairs))]

    probabilities, predictions, labels, metrics_dict = make_predictions(idx=idx,
                                                                        distances=test_distances,
                                                                        diagonal=diagonal,
                                                                        dummy_list=dummy_list)

    return probabilities, predictions, labels

# make_ppi_predictions()
