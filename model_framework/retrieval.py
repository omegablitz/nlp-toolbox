import pandas as pd
import logging
import random
import numpy as np
import re
import os
from framework import Task, ModelTrainer, FeatureEngineering, Evaluation, DataCuration
from rank_bm25 import BM25Okapi


class TaskRetrieval(Task):
    def __init__(self, config):
        self.config = config
    

class FeatureEngineeringRetrieval(FeatureEngineering):
    def __init__(self, data_args):
        self.data = data_args['dataset']
        self.data_args = data_args

    def tokenize_corpus(self, corpus):
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        return tokenized_corpus

    def split_doc(self, filename, split_size=100):
        # self.data[filename] -> Parsed OCR object
        # self.data.texts[filename] --> single string
        complete_texts = self.data.texts[filename]
        corpus = complete_texts.split("\n")
        # Better ideas to split
        # spaCy -> sentence splitter
        # para splitter
        # OCR cluster (white spacing) -- see  IbocrTextProcessing.cluster_based_on_DIST in framework.py
        return corpus

class Retrieval(ModelTrainer):
    def train(self, corpus, tokenized_corpus):
        self.corpus = corpus  # list of documents
        self.model = BM25Okapi(tokenized_corpus)

    def predict(self, query, len_results):
        tokenized_query = query.split(" ")
        doc_scores = self.model.get_scores(tokenized_query)
        # array([0.        , 0.93729472, 0.        ])

        return self.model.get_top_n(tokenized_query, self.corpus, n=len_results)

    def analyze_result(self, results):
        pass