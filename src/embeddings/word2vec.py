"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 April 2022
@Version  : 0.1
@Desc     : This file will be used to test classes and functions in development phase.
"""

from gensim.models import Word2Vec
from src.paths import *
import pandas as pd
import os
from src.paths import *
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from gensim.corpora.dictionary import Dictionary
from concurrent.futures import ProcessPoolExecutor




class Word2VecModel():
    """
    Class for word2vec with methods to calculate, save and load embeddings

    Args:
        doc_list: list: str
        dimension: int
    """
    def __init__(self, doc_list, dimension=100, model_path=None, calculate_embedding=False):
        self.embedding_initialized = False
        self.doc_list = doc_list
        self.dimension = dimension

        self.unseen = None
        self.doc_list = [[d for d in doc if d != "X"] for doc in self.doc_list]

        if model_path == None:
            self.model = Word2Vec(self.doc_list, window=5, vector_size=self.dimension, min_count=100, workers=7)
        else:
            self.model = Word2Vec.load(model_path)

        if calculate_embedding:
            self.calculate_embedding_distributed()


    def save_word2vec_model(self, save_path):

        self.model.save(save_path)


    def calculate_embedding(self):
        keys = set(self.model.wv.index_to_key)
        vec = []

        if self.unseen is not None:
            unseen_vec = self.model.wv.get_vector(self.unseen)

        for sentence in self.doc_list:
            if self.unseen is not None:
                vec.append(sum([self.model.wv.get_vector(y) if y in set(sentence) & keys
                                else unseen_vec for y in sentence]))
            else:
                vec.append(sum([self.model.wv.get_vector(y) for y in sentence
                            if y in set(sentence) & keys]))

        self.embedding = np.array(vec)
        self.embedding_initialized = True


    def vec_append_distributed(self, sentence, vec, keys):
        if self.unseen is not None:
            vec.append(sum([self.model.wv.get_vector(y) if y in set(sentence) & keys
                            else unseen_vec for y in sentence]))
        else:
            vec.append(sum([self.model.wv.get_vector(y) for y in sentence
                            if y in set(sentence) & keys]))


    def vec_append_distributed2(self, sentence):
        if self.unseen is not None:
            res = sum([self.model.wv.get_vector(y) if y in set(sentence) & self.keys
                            else self.unseen_vec for y in sentence])
        else:
            res = sum([self.model.wv.get_vector(y) for y in sentence
                            if y in set(sentence) & self.keys])

        return res



    def calculate_embedding_distributed(self):
        self.keys = set(self.model.wv.index_to_key)
        vec = []

        if self.unseen is not None:
            self.unseen_vec = self.model.wv.get_vector(self.unseen)

        # Parallel(n_jobs=7)(delayed(self.vec_append_distributed)(i, vec, keys) for i in tqdm(self.doc_list))
        with ProcessPoolExecutor(max_workers=4) as executor:
            for r in tqdm(executor.map(self.vec_append_distributed2, self.doc_list)):
                vec.append(r)

        self.embedding = np.array(vec)
        self.embedding_initialized = True


    def get_embedding(self):

        return self.embedding


    def save_embedding(self, file_name):
        if self.embedding_initialized:
            version = np.array([1])
            with open(file_name, "wb") as f:
                # Version
                np.save(f, version)
                np.save(f, self.embedding)


    @staticmethod
    def load_embedding(file_name):
        with open(file_name, "rb") as f:
            # Version
            version = np.load(f, allow_pickle=True)
            embedding = np.load(f, allow_pickle=True)

        return embedding

    @staticmethod
    def train(doc_list, model_name, dimension=100):
        word2vec_model = Word2VecModel(doc_list=doc_list, dimension=dimension)


        experiment_model_path = os.path.join(MODEL_DIRECTORY, model_name + str(dimension) + "_word2vec")
        if not os.path.exists(experiment_model_path):
            os.mkdir(experiment_model_path)

        word2vec_model.save_word2vec_model(
            os.path.join(experiment_model_path, model_name + "_" + str(dimension) + "_word2vec.model"))

    @staticmethod
    def train_kfold(doc_list, experiment_name, model_name, dimension=100):
        word2vec_model = Word2VecModel(doc_list=doc_list, dimension=dimension, calculate_embedding=False)


        experiment_model_path = os.path.join(MODEL_DIRECTORY, experiment_name)
        if not os.path.exists(experiment_model_path):
            os.mkdir(experiment_model_path)

        word2vec_model.save_word2vec_model(
            os.path.join(experiment_model_path, model_name + "_" + str(dimension) + "_word2vec_cosine.model"))



class Word2VecModel2():
    """
    Class for word2vec with methods to calculate, save and load embeddings

    Args:
        doc_list: list: str
        dimension: int
    """
    def __init__(self, doc_list, dimension=100, model_path=None):
        self.embedding_initialized = False
        self.doc_list = doc_list
        self.dimension = dimension

        self.unseen = None
        self.doc_list = [[d for d in doc if d != "X"] for doc in self.doc_list]

        self.dictionary = Dictionary(self.doc_list)
        self.bow = [self.dictionary.doc2bow(text) for text in self.doc_list]

        if model_path == None:
            self.model = Word2Vec(self.doc_list, window=5, vector_size=self.dimension, min_count=100, workers=7)
        else:
            self.model = Word2Vec.load(model_path)



    def save_word2vec_model(self, save_path):

        self.model.save(save_path)


    def calculate_embedding(self):
        keys = set(self.model.wv.index_to_key)
        vec = []

        if self.unseen is not None:
            unseen_vec = self.model.wv.get_vector(self.unseen)

        for sentence in self.doc_list:
            if self.unseen is not None:
                vec.append(sum([self.model.wv.get_vector(y) if y in set(sentence) & keys
                                else unseen_vec for y in sentence]))
            else:
                vec.append(sum([self.model.wv.get_vector(y) for y in sentence
                            if y in set(sentence) & keys]))

        self.embedding = np.array(vec)
        self.embedding_initialized = True


    def vec_append_distributed(self, sentence, vec, keys):
        if self.unseen is not None:
            vec.append(sum([self.model.wv.get_vector(y) if y in set(sentence) & keys
                            else unseen_vec for y in sentence]))
        else:
            vec.append(sum([self.model.wv.get_vector(y) for y in sentence
                            if y in set(sentence) & keys]))


    def vec_append_distributed2(self, sentence):
        if self.unseen is not None:
            res = sum([self.model.wv.get_vector(y) if y in set(sentence) & self.keys
                            else self.unseen_vec for y in sentence])
        else:
            res = sum([self.model.wv.get_vector(y) for y in sentence
                            if y in set(sentence) & self.keys])

        return res



    def calculate_embedding_distributed(self):
        self.keys = set(self.model.wv.index_to_key)
        vec = []

        if self.unseen is not None:
            self.unseen_vec = self.model.wv.get_vector(self.unseen)

        # Parallel(n_jobs=7)(delayed(self.vec_append_distributed)(i, vec, keys) for i in tqdm(self.doc_list))
        with ProcessPoolExecutor(max_workers=4) as executor:
            for r in tqdm(executor.map(self.vec_append_distributed2, self.doc_list)):
                vec.append(r)

        self.embedding = np.array(vec)
        self.embedding_initialized = True


    def get_embedding(self):

        return self.embedding


    def save_embedding(self, file_name):
        if self.embedding_initialized:
            version = np.array([1])
            with open(file_name, "wb") as f:
                # Version
                np.save(f, version)
                np.save(f, self.embedding)


    @staticmethod
    def load_embedding(file_name):
        with open(file_name, "rb") as f:
            # Version
            version = np.load(f, allow_pickle=True)
            embedding = np.load(f, allow_pickle=True)

        return embedding

    @staticmethod
    def train(doc_list, model_name, dimension=100):
        word2vec_model = Word2VecModel2(doc_list=doc_list, dimension=dimension)


        experiment_model_path = os.path.join(MODEL_DIRECTORY, model_name + str(dimension) + "_word2vec")
        if not os.path.exists(experiment_model_path):
            os.mkdir(experiment_model_path)

        word2vec_model.save_word2vec_model(
            os.path.join(experiment_model_path, model_name + "_" + str(dimension) + "_word2vec2.model"))

    @staticmethod
    def train_kfold(doc_list, experiment_name, model_name, dimension=100):
        word2vec_model = Word2VecModel2(doc_list=doc_list, dimension=dimension)


        experiment_model_path = os.path.join(MODEL_DIRECTORY, experiment_name)
        if not os.path.exists(experiment_model_path):
            os.mkdir(experiment_model_path)

        word2vec_model.save_word2vec_model(
            os.path.join(experiment_model_path, model_name + "_" + str(dimension) + "_word2vec_sc.model"))
