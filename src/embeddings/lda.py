"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 August 2022
@Version  : 0.1
@Desc     :
"""
import os

from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
import numpy as np
import pandas as pd
import os
from src.paths import *


class LDA():
    """
    Class for LDA with methods to calculate, save and load embeddings

    Args:
        doc_list: list: str
        data_type: str
        num_topics: int
    """
    def __init__(self, doc_list, num_topics, model_path=None):
        self.embedding_initialized = False
        self.doc_list = doc_list
        self.num_topics = num_topics

        self.docs_tokenised = [[d for d in doc.split(' ') if d] for doc in doc_list]
        self.docs_tokenised = [[d for d in doc if d != "X"] for doc in self.docs_tokenised]

        self.dictionary = Dictionary(self.docs_tokenised)

        self.bow = [self.dictionary.doc2bow(text) for text in self.docs_tokenised]
        if model_path == None:
            self.model = LdaModel(corpus=self.bow,
                                            num_topics=self.num_topics,
                                            id2word=self.dictionary)
        else:
            self.model = LdaModel.load(model_path)

        self.calculate_embedding()

    def save_lda_model(self, save_path):

        self.model.save(save_path)


    def calculate_embedding(self):
        self.embedding = np.zeros((len(self.bow), self.num_topics))

        for i in range(len(self.bow)):
            row = self.model.get_document_topics(self.bow[i])
            for topic, prob in row:
                self.embedding[i, topic] = prob

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
    def train(doc_list, model_name, dimension=10):
        lda_model = LDA(doc_list=doc_list, num_topics=dimension)


        experiment_model_path = os.path.join(MODEL_DIRECTORY, model_name + "_" + str(dimension) + "_lda")
        if not os.path.exists(experiment_model_path):
            os.mkdir(experiment_model_path)

        lda_model.save_lda_model(
            os.path.join(experiment_model_path, model_name + "_" + str(dimension) + "_lda.model"))

    @staticmethod
    def train_kfold(doc_list, experiment_name, model_name, dimension=10):
        lda_model = LDA(doc_list=doc_list, num_topics=dimension)


        experiment_model_path = os.path.join(MODEL_DIRECTORY, experiment_name)
        if not os.path.exists(experiment_model_path):
            os.mkdir(experiment_model_path)

        lda_model.save_lda_model(
            os.path.join(experiment_model_path, model_name + "_" + str(dimension) + "_lda.model"))

