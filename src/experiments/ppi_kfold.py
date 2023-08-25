import os
from sklearn.model_selection import KFold
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from gensim.models import TfidfModel

from src.utilities import *
from src.experiments.ppi_helpers import *
from src.paths import LISTS_DIRECTORY, MODEL_DIRECTORY, EMBEDDING_DIRECTORY
from src.experiments.ppi_helpers.ppi_functions import *
from src.embeddings.word2vec import Word2VecModel, Word2VecModel2
from src.embeddings.lda import LDA


n_splits = 10
list_file_path = os.path.join(LISTS_DIRECTORY, "pdb_list_5481.txt")
pdb_list = read_list(list_file_path)

indices = range(len(pdb_list))

kfold = KFold(n_splits=n_splits, random_state=42, shuffle=True)
shuffle = [(train.tolist(), test.tolist()) for train, test in kfold.split(indices)]
np.random.seed(seed=42)

class KfoldExperiments:
    @staticmethod
    def word2vec_cosine(shuffle):
        auc_list = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        for i, (train, test) in enumerate(shuffle):
            train_list = [pdb_list[i] for i in train]
            test_list = [pdb_list[i] for i in test]
            train_list, train_paires = make_doc_list(train_list)
            test_list, test_pairs = make_doc_list(test_list)

            experiment_name = f"{n_splits}fold_word2vec_cosine"
            model_name = f"fold_{i}"
            dimension = 100
            model_file_path = os.path.join(MODEL_DIRECTORY, experiment_name,
                                           model_name + "_" + str(dimension) + "_word2vec_cosine.model")
            if not os.path.isfile(model_file_path):
                Word2VecModel.train_kfold(train_list, experiment_name, model_name, dimension=dimension)
            word2vec_model = Word2VecModel(doc_list=test_list, dimension=dimension, model_path=model_file_path, calculate_embedding=True)
            experiment_embedding_directory = os.path.join(EMBEDDING_DIRECTORY, experiment_name)
            if not os.path.isdir(experiment_embedding_directory):
                os.mkdir(experiment_embedding_directory)
            word2vec_model.save_embedding(os.path.join(experiment_embedding_directory, "PDB_" + model_name + "_" + str(dimension) + "_word2vec_cosine.embeddding.npy"))
            embeddings = word2vec_model.embedding

            receptor_embeddings = embeddings[:test_pairs, :]
            ligand_embeddings = embeddings[test_pairs:, :]
            distance_array = cosine_similarity(receptor_embeddings, ligand_embeddings)

            diagonal = np.diagonal(distance_array)
            test_distances = remove_diag(distance_array)
            idx = [i for i in range(int(test_pairs))]

            dummy_list = []
            for id in idx:
                cidx = [j-1 for j in idx if j != id]
                random_idx = np.random.choice(cidx, 1, replace=True)[0]
                dummy_list.append(random_idx)

            w2v_probabilities, w2v_predictions, w2v_labels, metrics_dict = make_predictions_cosine(idx=idx,
                                                                                                   distances=test_distances,
                                                                                                   diagonal=diagonal,
                                                                                                   dummy_list=dummy_list)
            auc_list.append(metrics_dict["AUC"])
            accuracy_list.append(metrics_dict["Accuracy"])
            precision_list.append(metrics_dict["Precision"])
            recall_list.append(metrics_dict["Recall"])

        auc = sum(auc_list) / n_splits
        accuracy = sum(accuracy_list) / n_splits
        precision = sum(precision_list) / n_splits
        recall = sum(recall_list) / n_splits
        log_kfold_metrics(experiment_name=experiment_name,
                          n_splits=n_splits,
                          auc=auc,
                          accuracy=accuracy,
                          precision=precision,
                          recall=recall)

        return auc, accuracy, precision, recall

    @staticmethod
    def word2vec_sc(shuffle):
        auc_list = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        for i, (train, test) in enumerate(shuffle):
            train_list = [pdb_list[i] for i in train]
            test_list = [pdb_list[i] for i in test]
            train_list, train_paires = make_doc_list(train_list)
            test_list, test_pairs = make_doc_list(test_list)

            experiment_name = f"{n_splits}fold_word2vec_sc"
            model_name = f"fold_{i}"
            dimension = 100
            model_file_path = os.path.join(MODEL_DIRECTORY, experiment_name,
                                           model_name + "_" + str(dimension) + "_word2vec_sc.model")
            if not os.path.isfile(model_file_path):
                Word2VecModel2.train_kfold(train_list, experiment_name, model_name, dimension=dimension)
            word2vec_model = Word2VecModel2(doc_list=test_list, dimension=dimension, model_path=model_file_path)
            # experiment_embedding_directory = os.path.join(EMBEDDING_DIRECTORY, experiment_name)
            # if not os.path.isdir(experiment_embedding_directory):
            #     os.mkdir(experiment_embedding_directory)
            # word2vec_model.save_embedding(os.path.join(experiment_embedding_directory, "PDB_" + model_name + "_" + str(dimension) + "_word2vec_sc.embeddding.npy"))

            tfidf = TfidfModel(word2vec_model.bow)
            termsim_index = WordEmbeddingSimilarityIndex(word2vec_model.model.wv)
            termsim_matrix = SparseTermSimilarityMatrix(termsim_index, word2vec_model.dictionary, tfidf)

            distance_array = np.empty([test_pairs, test_pairs])
            for i in tqdm(range(test_pairs)):
                for j in range(test_pairs):
                    distance_array[i, j] = termsim_matrix.inner_product(tfidf[word2vec_model.bow[i]],
                                                                        tfidf[word2vec_model.bow[test_pairs + j]])

            diagonal = np.diagonal(distance_array)
            test_distances = remove_diag(distance_array)
            idx = [i for i in range(int(test_pairs))]

            dummy_list = []
            for id in idx:
                cidx = [j-1 for j in idx if j != id]
                random_idx = np.random.choice(cidx, 1, replace=True)[0]
                dummy_list.append(random_idx)

            w2v_probabilities, w2v_predictions, w2v_labels, metrics_dict = make_predictions_cosine(idx=idx,
                                                                                                   distances=test_distances,
                                                                                                   diagonal=diagonal,
                                                                                                   dummy_list=dummy_list)
            auc_list.append(metrics_dict["AUC"])
            accuracy_list.append(metrics_dict["Accuracy"])
            precision_list.append(metrics_dict["Precision"])
            recall_list.append(metrics_dict["Recall"])

        auc = sum(auc_list) / n_splits
        accuracy = sum(accuracy_list) / n_splits
        precision = sum(precision_list) / n_splits
        recall = sum(recall_list) / n_splits
        log_kfold_metrics(experiment_name=experiment_name,
                          n_splits=n_splits,
                          auc=auc,
                          accuracy=accuracy,
                          precision=precision,
                          recall=recall)

        return auc, accuracy, precision, recall

    @staticmethod
    def lda(shuffle):
        auc_list = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        for i, (train, test) in enumerate(shuffle):
            train_list = [pdb_list[i] for i in train]
            test_list = [pdb_list[i] for i in test]
            train_list, train_pairs = make_doc_list(train_list)
            test_list, test_pairs = make_doc_list(test_list)

            experiment_name = f"{n_splits}fold_lda"
            model_name = f"fold_{i}"
            dimension = 10
            model_file_path = os.path.join(MODEL_DIRECTORY, experiment_name,
                                           model_name + "_" + str(dimension) + "_lda.model")
            if not os.path.isfile(model_file_path):
                LDA.train_kfold(train_list, experiment_name, model_name, dimension=dimension)
            lda_model = LDA(doc_list=test_list, num_topics=dimension, model_path=model_file_path)
            experiment_embedding_directory = os.path.join(EMBEDDING_DIRECTORY, experiment_name)
            if not os.path.isdir(experiment_embedding_directory):
                os.mkdir(experiment_embedding_directory)
            lda_model.save_embedding(os.path.join(experiment_embedding_directory, "PDB_" + model_name + "_" + str(dimension) + "_lda.embeddding.npy"))
            embeddings = lda_model.embedding

            distance_array = np.empty([test_pairs, test_pairs])
            receptor_embeddings = embeddings[:test_pairs, :]
            ligand_embeddings = embeddings[test_pairs:, :]
            for i in tqdm(range(int(test_pairs))):
                distance_array[i, :] = bhattacharyya_dist(receptor_embeddings[i, :], ligand_embeddings)

            diagonal = np.diagonal(distance_array)
            test_distances = remove_diag(distance_array)
            idx = [i for i in range(int(test_pairs))]

            dummy_list = []
            for id in idx:
                cidx = [j-1 for j in idx if j != id]
                random_idx = np.random.choice(cidx, 1, replace=True)[0]
                dummy_list.append(random_idx)

            lda_probabilities, lda_predictions, lda_labels, metrics_dict = make_predictions(idx=idx,
                                                                                            distances=test_distances,
                                                                                            diagonal=diagonal,
                                                                                            dummy_list=dummy_list)
            auc_list.append(metrics_dict["AUC"])
            accuracy_list.append(metrics_dict["Accuracy"])
            precision_list.append(metrics_dict["Precision"])
            recall_list.append(metrics_dict["Recall"])

        auc = sum(auc_list) / n_splits
        accuracy = sum(accuracy_list) / n_splits
        precision = sum(precision_list) / n_splits
        recall = sum(recall_list) / n_splits
        log_kfold_metrics(experiment_name=experiment_name,
                          n_splits=n_splits,
                          auc=auc,
                          accuracy=accuracy,
                          precision=precision,
                          recall=recall)

        return auc, accuracy, precision, recall

KfoldExperiments.lda(shuffle)
KfoldExperiments.word2vec_sc(shuffle)
KfoldExperiments.word2vec_cosine(shuffle)
