from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from gensim.models import TfidfModel

from src.embeddings.word2vec import Word2VecModel, Word2VecModel2
from src.paths import *
from src.utilities import *
from src.experiments.ppi_helpers.ppi_functions import *

MAKE_DUMMY_LIST = False

# Seed script
np.random.seed(42)

def make_ppi_predictions_word2vec_cosine():
    dummy_list = read_list(os.path.join(LISTS_DIRECTORY, "dummy_id.txt"))
    dimension = 100

    model_name = "ppi_pred"
    model_file_path = os.path.join(MODEL_DIRECTORY, model_name + str(dimension) + "_word2vec",
                                   model_name + "_" + str(dimension) + "_word2vec.model")
    if not os.path.isfile(model_file_path):
        train_list = read_list(os.path.join(LISTS_DIRECTORY, "train.txt"))
        train_list, pairs = make_doc_list(train_list)
        Word2VecModel.train(train_list, model_name, dimension=dimension)

    text_list = read_list(os.path.join(LISTS_DIRECTORY, "test.txt"))
    text_list, pairs = make_doc_list(text_list)
    word2vec_model = Word2VecModel(doc_list=text_list, dimension=dimension, model_path=model_file_path, calculate_embedding=True)
    word2vec_model.save_embedding(os.path.join(EMBEDDING_DIRECTORY, "PDB_" + model_name + "_" + str(dimension) + "_word2vec.embeddding.npy"))
    embeddings = word2vec_model.embedding

    receptor_embeddings = embeddings[:pairs, :]
    ligand_embeddings = embeddings[pairs:, :]
    distance_array = cosine_similarity(receptor_embeddings, ligand_embeddings)


    diagonal = np.diagonal(distance_array)
    test_distances = remove_diag(distance_array)
    idx = [i for i in range(int(pairs))]

    probabilities, predictions, labels, metrics_dict = make_predictions_cosine(idx=idx,
                                                                               distances=test_distances,
                                                                               diagonal=diagonal,
                                                                               dummy_list=dummy_list)

    return probabilities, predictions, labels



def make_ppi_predictions_word2vec_sc():
    dummy_list = read_list(os.path.join(LISTS_DIRECTORY, "dummy_id.txt"))
    dimension = 100

    model_name = "ppi_pred"
    model_file_path = os.path.join(MODEL_DIRECTORY, model_name + str(dimension) + "_word2vec",
                                   model_name + "_" + str(dimension) + "_word2vec2.model")
    if not os.path.isfile(model_file_path):
        train_list = read_list(os.path.join(LISTS_DIRECTORY, "train.txt"))
        train_list, pairs = make_doc_list(train_list)
        Word2VecModel2.train(train_list, model_name, dimension=dimension)

    text_list = read_list(os.path.join(LISTS_DIRECTORY, "test.txt"))
    text_list, pairs = make_doc_list(text_list)
    word2vec_model = Word2VecModel2(doc_list=text_list, dimension=dimension, model_path=model_file_path)

    tfidf = TfidfModel(word2vec_model.bow)
    termsim_index = WordEmbeddingSimilarityIndex(word2vec_model.model.wv)
    termsim_matrix = SparseTermSimilarityMatrix(termsim_index, word2vec_model.dictionary, tfidf)

    distance_array = np.empty([pairs, pairs])
    for i in tqdm(range(pairs)):
        for j in range(pairs):
            distance_array[i, j] = termsim_matrix.inner_product(tfidf[word2vec_model.bow[i]], tfidf[word2vec_model.bow[pairs + j]])

    diagonal = np.diagonal(distance_array)
    test_distances = remove_diag(distance_array)
    idx = [i for i in range(int(pairs))]

    probabilities, predictions, labels, metrics_dict = make_predictions_cosine(idx=idx,
                                                                               distances=test_distances,
                                                                               diagonal=diagonal,
                                                                               dummy_list=dummy_list)

    return probabilities, predictions, labels