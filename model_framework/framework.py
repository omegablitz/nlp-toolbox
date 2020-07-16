import csv
import json
import os
import pandas as pd
import re
import string
import logging
import numpy as np
import math

from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import BertModel
from transformers import BertTokenizer
import numpy as np
import torch
from torch import nn
import logging
import itertools
import difflib

# Point to instabase SDK, use SDK to download
import sys
sys.path.append('/Users/ahsaasbajaj/Documents/Code/instabase/sdk/src/py')
import importlib
import instabase_sdk
importlib.reload(instabase_sdk)
from instabase_sdk import Instabase

PUNC_TABLE = str.maketrans({key: None for key in string.punctuation})

from test_classifier import infer_classifier

# Import instabase
import sys
sys.path.append(
    '/Users/ahsaasbajaj/Documents/Code/instabase/distributed-tasks/celery/app-tasks/build/py'
)
from instabase.ocr.client.libs.ibocr import ParsedIBOCRBuilder


class DatasetWarning():
    def __init__(self, warning_type, message):
        self.warning_type = warning_type
        self.message = message
    def __repr__(self):
        return 'Warning[{}, {}]'.format(self.warning_type, self.message)

class Usecase():
    def __init__(self, usecase):
        self.usecase = usecase
    
    def __repr__(self):
        return 'Usecase[{}, {}, {}]'.format(self.usecase, self.dataset)

    def setConfig(self, config):
        self.config = config
        self.setDataset()
        self.setLabelsDict()
        self.setCatDict()

    def setDataset(self):
      self.dataset = self.config['dataset']

    def setLabelsDict(self):
        self.labels_dict = self.config['labels_dict']
    
    def setCatDict(self):
        self.cat_dict = self.config['cat_dict']


class StringProcessing():
    @staticmethod
    def isSafe(row):
        # retunrs True if string doesn't contain any numbers
        if any(char.isdigit() for char in row):
            return False
        if len(row) < 5:
            # string less than 5 characters or single word
            return False
        return True
    
    @staticmethod
    def find_remove_opening_token(row, token):
        # remove quotes from opening
        indx = row.find(token)
        if indx == 0:
            row = row[1:]
        return row

    @staticmethod
    def clean_string(strng):
        names = strng.split(" ")
        ans = []
        for p in names:
            if p != '':
                ans.append(p.lower().strip())
                
        result = " ".join(ans)
        result = result.lower().strip()
        return result

class DataCuration():

  def __init__(self, access_token, dataset_paths, dataset_config,
               goldens_paths, goldens_config):
    self.ib = Instabase('https://instabase.com', access_token)
    self.dataset_config = dataset_config
    self.golden_config = goldens_config

    if type(dataset_paths) != list:
      dataset_paths = [dataset_paths]

    if type(goldens_paths) != list:
      goldens_paths = [goldens_paths]

    self.datadir = dataset_paths
    self._load_ib_dataset(dataset_paths, dataset_config)
    self._load_goldens(goldens_paths, goldens_config)
    if dataset_config['convert2txt']:
        self.processIBOCR2txt(dataset_config)

  def _load_goldens(self, goldens_paths, goldens_config):
    goldens_type = goldens_config.get('file_type')
    for goldens_path in goldens_paths:
      logging.info('Loading goldens from {}'.format(goldens_path))
      if goldens_type == 'csv':
        golden_all_df = pd.read_csv(goldens_path)
        self.golden_all = golden_all_df.set_index(goldens_config['index_field_name'])

        # filter goldens by presence in dataset
        def ispresent(row):
            if row in self.dataset:
                return True
            return False
        golden_all_df['in_src'] = golden_all_df[goldens_config['index_field_name']].apply(ispresent)
        golden_df = golden_all_df[golden_all_df['in_src'] == True]
        golden_df = golden_df.drop('in_src', axis=1)
        self.golden = golden_df.set_index(goldens_config['index_field_name'])

        print('Total files Goldens: ', self.golden_all.shape)
        print('Total files found in the source', self.golden.shape)

  def _load_ib_dataset(self, dataset_paths, dataset_config):
    self.dataset = {}
    for dataset_path in dataset_paths:
      logging.info('Loading dataset from {}'.format(dataset_path))
      if dataset_config['is_local'] is False:
          file_result = self.ib.list_dir(dataset_path)
          files = [node['full_path'] for node in file_result['nodes']]
          file_objects = []
          for file in files:
              with open(file) as f:
                file_objects.append(self.ib.read_file(f))
      else:
          files = os.listdir(dataset_path)
          file_objects = []
          for file in files:
              with open(os.path.join(dataset_path, file)) as f:
                file_objects.append(f.read())

      for file, file_object in zip(files, file_objects):
        content = None
        identifier = file
        if dataset_config.get('file_type') in ['ibdoc', 'ibocr']:
          ibocr, err = ParsedIBOCRBuilder.load_from_str(os.path.join(dataset_path, file), file_object)
          content = ibocr.as_parsed_ibocr()
        if dataset_config.get('identifier'):
          identifier = dataset_config.get('identifier')(file)
        self.dataset.update({identifier: content})

  def processIBOCR2txt(self, dataset_config):
    #Todo: add reading from ib
    self.texts = {}
    for data_dir in self.datadir:
        files = os.listdir(data_dir)
        print("Processing {} IBOCR files to txt".format(len(files)))

        for fname in files:
            fpath = os.path.join(data_dir, fname)
            f = open(fpath)
            file = json.load(f)
            dictionary = file[0]
            dictionary.keys()
            texts = dictionary['text']
            if dataset_config.get('identifier'):
                identifier = dataset_config.get('identifier')(fname)
            self.texts.update({identifier: texts})

  def processIBOCR2candidatePhrases(self, dataset_config, processing_config):
    #Todo: add reading from ib

    self.candidates = {}
    X_DIST_THRESHOLD = processing_config['X_DIST_THRESHOLD']
    for data_dir in self.datadir:
        files = os.listdir(data_dir)
        print("Generating candidates for {} files".format(len(files)))
        for fname in files:
            fpath = os.path.join(data_dir, fname)
            f = open(fpath)
            file = json.load(f)
            dictionary = file[0]
            dictionary.keys()
            lines = dictionary['lines']

            words = [[]]
            for line in lines:
                for word in line: 
                    start = word['start_x']
                    if not words[-1]:
                        words[-1].append(word)
                        continue
                    if start - words[-1][-1]['end_x'] < X_DIST_THRESHOLD:
                        words[-1].append(word)
                        continue
                    words.append([word])
                words.append([])
            
            phrases = []
            for cluster in words:
                phrases.append(' '.join([w['word'] for w in cluster]))
        
            # remove special characters and other filters
            candidates = []
            for phrase in phrases:
                phrase = StringProcessing.find_remove_opening_token(phrase, '"')
                if StringProcessing.isSafe(phrase):
                    candidates.append(phrase)

            if dataset_config.get('identifier'):
                identifier = dataset_config.get('identifier')(fname)
            self.candidates.update({identifier: candidates})

  def compare_candidates_and_goldens(self, processing_config, candidates_fields):
        # goldens may have more keys than dataset
        filenames = []
        person_found = 0
        org_found = 0
        person_found_files = []
        org_found_files = []

        for key in self.candidates:
            phrases = self.candidates[key]     
            cds = set([StringProcessing.clean_string(elt) for elt in phrases])
            if key not in self.golden.index:
                continue

            this_person = StringProcessing.clean_string(self.golden.loc[key, candidates_fields['person']])
            this_org = StringProcessing.clean_string(self.golden.loc[key, candidates_fields['org']])
            
            if this_person in cds:
                person_found += 1
                person_found_files.append(key)
            if this_org in cds:
                org_found += 1
                org_found_files.append(key)
            
        #     if this_person not in cds:
        #         print(key)
        #         print(this_person, difflib.get_close_matches(this_person, cds), '\n')
            # if this_org not in cds:
            #     print(key)
            #     print(this_org, difflib.get_close_matches(this_org, cds), '\n')
            filenames.append(key)


        total_files = len(filenames)
        print("For X_DIST_THRESHOLD configuraion: {0}".format(processing_config['X_DIST_THRESHOLD']))
        print("total files: {0}\nperson names found in candidates: {1}\norg names found in candidates: {2}\n".format(total_files, len(person_found_files), len(org_found_files)))
    

class FeatureEngineering():
    def __init__(self, usecase, data_curation, candidates_fields):
        self.labels_dict = usecase.labels_dict
        self.data = data_curation
        self.candidates_fields = candidates_fields

    def generate_test_samples(self):
        return

    def generate_test_samples_from_goldens(self):
        labels = []
        contexts = []

        persons = self.data.golden[self.candidates_fields['person']].tolist()
        person_label = [self.labels_dict['person']] * len(persons)

        contexts.extend(persons)
        labels.extend(person_label)

        org = self.data.golden[self.candidates_fields['org']].tolist()
        org_label = [self.labels_dict['org']] * len(org)

        contexts.extend(org)
        labels.extend(org_label)

        test_samples = pd.DataFrame(list(zip(contexts, labels)), columns=['context', 'label'])
        test_samples = test_samples.sample(frac=1)
        test_samples = test_samples.dropna()
        return test_samples

    def generate_test_samples_from_candidates(self):
        test_samples = {}
        for key in self.data.candidates:
            candidate_list = self.data.candidates[key]
            labels = [np.nan] * len(candidate_list)
            df = pd.DataFrame(list(zip(candidate_list, labels)), columns=['context', 'label'])
            df = df.drop_duplicates()  # redundant strings
            df['context'] = df['context'].astype(str)
            df['label'] = df['label'].astype(float)
            test_samples[key] = df
        return test_samples


class ModelEvaluator():
    def __init__(self, usecase):
        self.labels_dict = usecase.labels_dict
        self.cat_dict = usecase.cat_dict

    def set_config(self, model_config, eval_config):
        self.model_config = model_config
        self.eval_config = eval_config

    def getRetrRelSet(self, retrieved, relevant):
        if isinstance(retrieved, list):
            retrieved_set = set(retrieved) # model (recommendations)
        elif isinstance(retrieved, str):
            # set with one element
            retrieved_set = {retrieved} 
        elif isinstance(retrieved, float):
            # empty dataframe cell
            retrieved_set = {}

        if isinstance(relevant, list):
            relevant_set = set(relevant) # gold (saw)
        elif isinstance(relevant, str):
            relevant_set = {relevant}
        elif isinstance(relevant, float):
            relevant_set = {}
            
        return retrieved_set, relevant_set

    def getPrecisionRecall(self, retrieved, relevant):
        # do lower() and strip() here
        # "recommended movies saw by the user" => retrieved.intersection(relevant)
        # To get the "number of recommended movie that the user saw":
        
        
        clean_retrieved = set([StringProcessing.clean_string(r) for r in retrieved])        
        clean_relevant = set([StringProcessing.clean_string(r) for r in relevant])
        
        intersect = len(clean_retrieved.intersection(clean_relevant))

        if len(clean_retrieved) == 0:
            precision = 1
        else:
            precision = intersect/len(clean_retrieved)

        if len(clean_relevant) == 0:
            # some goldens don't have company names in Resume
            recall = 1
        else:
            recall = intersect/len(clean_relevant)
        
        return precision, recall

    def run_evaluation(self, testdata):
        model_file_or_path = self.model_config['model_file_or_path']
        self.num_labels = self.model_config['num_labels']
        gpu = self.eval_config['gpu']
        if self.eval_config['use_goldens']: 
            # testdata is single dataframe as data is generated using goldens csv
            return infer_classifier(model_file_or_path, testdata, self.num_labels, gpu)
        else:
            # test_data is a dictionary {'filename' : dataframe}
            results = {}
            for key in testdata:
                results[key] = infer_classifier(model_file_or_path, testdata[key], self.num_labels, gpu)
            return results

    def analyze_golden_result(self, results):
        labels = [x for x in range(self.num_labels - 1)] # Not to include 'None' class since it is never in true_label
        true_labels = results['label']
        predictions = results['predicted']

        # Overall Score
        micro_precision = precision_score(true_labels, predictions, labels=labels, average="micro")
        micro_recall = recall_score(true_labels, predictions,  labels=labels, average="micro")
        micro_f1 = f1_score(true_labels, predictions, labels=labels, average="micro")
        print('Micro Test R: {0:0.4f}, P: {1:0.4f}, F1: {2:0.4f}'.format(micro_recall, micro_precision, micro_f1))

        macro_precision = precision_score(true_labels, predictions, labels=labels, average="macro")
        macro_recall = recall_score(true_labels, predictions, labels=labels, average="macro")
        macro_f1 = f1_score(true_labels, predictions, labels=labels, average="macro")
        print('Macro Test R: {0:0.4f}, P: {1:0.4f}, F1: {2:0.4f}'.format(macro_recall, macro_precision, macro_f1))

        # Class Wise Score
        for lab in labels:
            # micro vs macro doesn't matter in case of single-label
            precision = precision_score(true_labels, predictions, labels=[lab], average="micro")
            recall = recall_score(true_labels, predictions,  labels=[lab], average="micro")
            f1 = f1_score(true_labels, predictions, labels=[lab], average="micro")
            print('Category: {0}, Test R: {1:0.4f}, P: {2:0.4f}, F1: {3:0.4f}'.format(self.cat_dict[lab], recall, precision, f1))
    
    def analyze_overall_result(self, results, goldens, candidates_fields):
        # results is a list of dataframes (one for each file) -- BERT's outputs
        person_recall = []
        person_precision = []
        org_recall = []
        org_precision = []

        for key in results:
            bert_results = results[key]
            bert_person_df = bert_results[bert_results['predicted'] == self.labels_dict['person']]
            bert_org_df = bert_results[bert_results['predicted'] == self.labels_dict['org']]
            
            retrieved_person = bert_person_df['context'].tolist()
            retrieved_org = bert_org_df['context'].tolist()
            relevant_person = goldens.loc[key, candidates_fields['person']]
            relevant_org = goldens.loc[key, candidates_fields['org']]
            
            # PERSON NAMES
            retrieved, relevant = self.getRetrRelSet(retrieved_person, relevant_person)
            precision, recall = self.getPrecisionRecall(retrieved, relevant)
            person_precision.append(precision)
            person_recall.append(recall)

            # ORG NAMES
            retrieved, relevant = self.getRetrRelSet(retrieved_org, relevant_org)
            precision, recall = self.getPrecisionRecall(retrieved, relevant)
            org_precision.append(precision)
            org_recall.append(recall)
    
        r = np.mean(person_recall)
        p = np.mean(person_precision)
        f1 = 2*p*r/(p + r)
        print("For field {0}, recall: {1:0.4f}, precision: {2:0.4f}, F1: {3:0.4f} ".format('person', r, p, f1))

        r = np.mean(org_recall)
        p = np.mean(org_precision)
        f1 = 2*p*r/(p + r)
        print("For field {0}, recall: {1:0.4f}, precision: {2:0.4f}, F1: {3:0.4f} ".format('org', r, p, f1))