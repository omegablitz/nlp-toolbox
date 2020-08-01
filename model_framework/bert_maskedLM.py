import pandas as pd
import logging
import random
import numpy as np
import re
import os
from framework import Task, ModelTrainer, FeatureEngineering, Evaluation, DataCuration
from infer_bert_maskedLM import get_lm_inference


class BERTMaskedLM(ModelTrainer):
    def train(self):
        pass

    def predict(self, queries):
        model_file_or_path = self.training_args['model_file_or_path']
        gpu = self.training_args['gpu']
        output_dir = self.training_args['output_dir']
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        get_lm_inference(queries, model_file_or_path, output_dir, gpu)

    def analyze_result(self, results):
        pass

