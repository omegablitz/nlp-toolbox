import pandas as pd
import logging
import random
import numpy as np
import re
import os
from framework import Task, ModelTrainer, FeatureEngineering, Evaluation, DataCuration
from infer_bert_qa import get_qa_inference


class TaskQA(Task):
    def __init__(self, config):
        self.config = config
    

class FeatureEngineeringQA(FeatureEngineering):
    def __init__(self, data_args):
        self.data = data_args['dataset']
        self.data_args = data_args

class BERTQA(ModelTrainer):
    def train(self):
        pass

    def predict(self, queries):
        model_file_or_path = self.training_args['model_file_or_path']
        gpu = self.training_args['gpu']
        output_dir = self.training_args['output_dir']
        data_texts = self.data_args['dataset'].texts

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if isinstance(queries, str) is True:
            queries = [queries]

        return get_qa_inference(data_texts, queries, model_file_or_path, output_dir, gpu)

    def analyze_result(self, results):
        pass

