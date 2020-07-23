import csv
import json
import os
import pandas as pd
import re
import string
import logging
import numpy as np
import math
import ast

from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import torch
from torch import nn
import logging
import itertools
import difflib
import random
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Point to instabase SDK, use SDK to download. We only need this for development, can remove this later 
# ToDo: This is fine for now, but we should make it so that this is an ENV variable (or eventually an actual pip dependency)
import sys
sys.path.append('/Users/ahsaasbajaj/Documents/Code/instabase/sdk/src/py')
import importlib
import instabase_sdk
importlib.reload(instabase_sdk)
from instabase_sdk import Instabase

PUNC_TABLE = str.maketrans({key: None for key in string.punctuation})

from infer_bert_classifier import get_classifier_inference
from bert_utils import update_bert_embeddings
from rule_features import rules
from preprocessing import preprocessing_rules

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# Import instabase. We only need this for development, can remove this later 
# ToDo: This is fine for now, but we should make it so that this is an ENV variable (or eventually an actual pip dependency)
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

class Task():
     def __init__(self, config):
        self.config = config

class Task_NER():
    def __init__(self, config):
        self.config = config
        self.set_labels_dict()

    def set_labels_dict(self):
        self.labels_dict = self.config['labels_dict']
        self.cat_dict = {v: k for k, v in self.labels_dict.items()}

class StringProcessing():
    @staticmethod
    def remove_digits_filter_by_length(row, min_len):
        # retunrs True if string doesn't contain any numbers
        if any(char.isdigit() for char in row):
            return False
        if len(row) < min_len:
            # string less than min_len characters or single word
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

    @staticmethod
    def clean_token(token):
        return token.lower().translate(PUNC_TABLE)

class IbocrTextProcessing():
    @staticmethod
    def process_IBOCR_to_txt(all_data_dirs, dataset_config):
        #ToDo: check reading from ib, and for ibdoc
        texts = {}
        for data_dir in all_data_dirs:
            files = os.listdir(data_dir)
            logging.info("Processing {} IBOCR files to txt".format(len(files)))

            for fname in files:
                fpath = os.path.join(data_dir, fname)
                f = open(fpath)
                file = json.load(f)
                dictionary = file[0]
                dictionary.keys()
                texts_list = dictionary['text']
                if dataset_config.get('identifier'):
                    identifier = dataset_config.get('identifier')(fname)
                texts.update({identifier: texts_list})
        
        return texts

    @staticmethod
    def cluster_based_on_DIST(lines, threshold):
        words = [[]]
        for line in lines:
            for word in line: 
                start = word['start_x']
                if not words[-1]:
                    words[-1].append(word)
                    continue
                if start - words[-1][-1]['end_x'] < threshold:
                    words[-1].append(word)
                    continue
                words.append([word])
            words.append([])
        
        phrases = []
        for cluster in words:
            phrases.append(' '.join([w['word'] for w in cluster]))

        return phrases

    @staticmethod
    def process_IBOCR_to_candidate_phrases(all_data_dirs, dataset_config, processing_config):
        #ToDo: check reading from ib, and for ibdoc

        candidates = {}
        # ToDo: Eventually (or now, not sure) we would also want to cluster based on Y distance. 
        # This clustering approach may be good to separate out into a separate file that has a bunch of "algorithms" for text / IBOCR processing
        for data_dir in all_data_dirs:
            files = os.listdir(data_dir)
            logging.info("Generating candidates for {} files".format(len(files)))
            for fname in files:
                fpath = os.path.join(data_dir, fname)
                f = open(fpath)
                file = json.load(f)
                dictionary = file[0]
                dictionary.keys()
                lines = dictionary['lines']
                phrases = IbocrTextProcessing.cluster_based_on_DIST(lines, threshold=processing_config['X_DIST_THRESHOLD'])

                # remove special characters and other filters
                candidate_list = []
                for phrase in phrases:
                    phrase = StringProcessing.find_remove_opening_token(phrase, '"')
                    if StringProcessing.remove_digits_filter_by_length(phrase, min_len=5):
                        candidate_list.append(phrase)

                if dataset_config.get('identifier'):
                    identifier = dataset_config.get('identifier')(fname)
                candidates.update({identifier: candidate_list})
        
        return candidates


class DataCuration():

    def __init__(self, access_token, dataset_config, goldens_config):
        # ToDo: Ideally, we should make this configurable, since sometimes we will want instabase.com, dogfood, a custom URL at a customer site, etc.
        self.ib = Instabase('https://instabase.com', access_token)
        self.dataset_config = dataset_config
        self.golden_config = goldens_config

        dataset_paths = dataset_config['path']
        goldens_paths = goldens_config['path']

        if type(dataset_paths) != list:
            dataset_paths = [dataset_paths]

        if type(goldens_paths) != list:
            goldens_paths = [goldens_paths]

        self.datadir = dataset_paths
        self._load_ib_dataset(dataset_paths, dataset_config)
        self._load_goldens(goldens_paths, goldens_config)

        if dataset_config['convert2txt']:
            self.texts = IbocrTextProcessing.process_IBOCR_to_txt(dataset_paths, dataset_config)

    def get_file_objects(self, dataset_path, read_from_local):
        file_objects = []
        if read_from_local is False:
                file_result = self.ib.list_dir(dataset_path)
                files = [node['full_path'] for node in file_result['nodes']]
                for file in files:
                    file_objects.append(self.ib.read_file(file))
        else:
            files = os.listdir(dataset_path)
            for file in files:
                with open(os.path.join(dataset_path, file)) as f:
                    file_objects.append(f.read())
        
        return files, file_objects

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

                if len(list(self.golden.index)) != len(set(self.golden.index)):
                    logging.info("Goldens have non-unique filenames, keeping only the first values")
                    self.golden = self.golden.loc[~self.golden.index.duplicated(keep='first')]

                logging.info('Total files Goldens: {}'.format(self.golden_all.shape))
                logging.info('Total files found in the source with unique index: {}'.format(self.golden.shape))

    def _load_ib_dataset(self, dataset_paths, dataset_config):
        self.dataset = {}
        files = []
        file_objects = []

        for dataset_path in dataset_paths:
            logging.info('Loading dataset from {}'.format(dataset_path))
            this_files, this_file_objects = self.get_file_objects(dataset_path, read_from_local=dataset_config['is_local'])
            files.extend(this_files)
            file_objects.extend(this_file_objects)
            logging.info("{} files loaded".format(len(files)))

        for file, file_object in zip(files, file_objects):
            content = None
            identifier = file
            if dataset_config.get('file_type') in ['ibdoc', 'ibocr']:
                ibocr, err = ParsedIBOCRBuilder.load_from_str(os.path.join(dataset_path, file), file_object)
                content = ibocr.as_parsed_ibocr()
                if dataset_config.get('identifier'):
                    identifier = dataset_config.get('identifier')(file)
                    self.dataset.update({identifier: content})

    def generate_candidates_phrases(self, processing_config):
        self.candidates = IbocrTextProcessing.process_IBOCR_to_candidate_phrases(self.datadir, self.dataset_config, processing_config)
        self.processing_config = processing_config

    def compare_candidates_and_goldens(self, candidates_fields):
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
        #         logging.debug(key)
        #         logging.debug(this_person, difflib.get_close_matches(this_person, cds), '\n')
            # if this_org not in cds:
            #     logging.debug(key)
            #     logging.debug(this_org, difflib.get_close_matches(this_org, cds), '\n')
            filenames.append(key)


        total_files = len(filenames)
        logging.info("For X_DIST_THRESHOLD configuraion: {0}".format(self.processing_config['X_DIST_THRESHOLD']))
        logging.info("total files: {0}\nperson names found in candidates: {1}\norg names found in candidates: {2}\n".format(total_files, len(person_found_files), len(org_found_files)))
    

    def generate_spatial_samples(self, field_to_capture, data_config):
        """
        Generates samples to train a model on, based on contextual information
        surrounding a token.
        """
        
        samples = []
        targets = []
        warnings = []

        for sample_name in list(self.golden.index):
            if sample_name not in self.dataset:
                warnings.append(DatasetWarning('SampleNotFound', 'Golden file "{}" not found in dataset'.format(sample_name)))
                continue
                
            ibdoc = self.dataset[sample_name].get_joined_page()[0]
            featurizer = IBDOCFeaturizer(ibdoc)
            
            expected_field = self.golden.at[sample_name, field_to_capture]

            if isinstance(expected_field, str):
                if len(expected_field.strip()) == 0:
                    warnings.append(DatasetWarning('FieldEmpty', 'Golden file "{}" has no entry for "{}"'.format(sample_name, field_to_capture)))
                    continue
            else:
                # expected_fiels may contain NaN
                warnings.append(DatasetWarning('FieldEmpty', 'Golden file "{}" has no entry for "{}"'.format(sample_name, field_to_capture)))
                continue

            # First, collect a number of samples relevant for the task
            expected_tokens = [StringProcessing.clean_token(w) for w in expected_field.split()]
            expected = ' '.join(expected_tokens)
            all_tokens = featurizer.get_all_tokens()
            found_indices = set()
            for i in range(len(all_tokens) - len(expected_tokens)):
                token_range = all_tokens[i:i+len(expected_tokens)]
                tokens_to_compare = [StringProcessing.clean_token(w['word']) for w in token_range]
                if ' '.join(tokens_to_compare) == expected:
                    for idx in range(i, i+len(expected_tokens)):
                        found_indices.add(idx)
            if not found_indices:
                warnings.append(DatasetWarning('TargetNotFound', 'Not able to locate field "{}" in file "{}"'.format(field_to_capture, sample_name)))
                continue

            # Collect negative examples
            if not data_config['balance_targets']:
                print("NOT BALANCING TARGETS")
                fvs = featurizer.get_feature_vectors(data_config)
                for idx in range(len(fvs)):
                    samples.append(fvs[idx])
                    targets.append(1 if idx in found_indices else 0)
            else:
                negative_examples = list(set(range(len(featurizer.get_all_tokens()))) - found_indices)
                selected_negative_examples = np.random.choice(negative_examples, len(found_indices), replace=False)
                fvs = []
                for idx in (list(found_indices) + list(selected_negative_examples)):
                    fv = featurizer.get_token_feature_vector(idx, data_config)
                    samples.append(fv)
                    targets.append(1 if idx in found_indices else 0)


        return (np.array(samples), np.array(targets), warnings)


class EmbeddingCache:
    def __init__(self):
        # Model cache
        self.glove_model = None
        self.bert_model = None

        # Result cache
        self.glove = {}
        self.bert = {}

# Cache for word embeddings
EMBEDDING_CACHE = EmbeddingCache()

class ModelTrainer():
    def __init__(self, training_args):
        self.training_args = training_args

class MLP(ModelTrainer):
    def train(self, X_train, X_test, y_train, y_test):
        # Neural network
        model = Sequential()
        model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        logging.info('Training multilayer perceptron model for {} samples'.format(X_train.shape[0]))
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
            epochs=self.training_args['epochs'], batch_size=self.training_args['batch_size'])

        self.model = model
        self.history = history

    def evaluate(self):
        logging.info(self.history.history.keys())
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss']) 
        plt.title('Model loss') 
        plt.ylabel('Loss') 
        plt.xlabel('Epoch') 
        plt.legend(['Train', 'Test'], loc='upper left') 
        plt.show()
        return self.history.history['val_accuracy'][-1]


class FeatureEngineering():
    @staticmethod
    def product_dict(**kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    @staticmethod
    def dist(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    @staticmethod
    def get_subset_for_debugging(data_object, sample_size=10):
        if isinstance(data_object, dict):
            return dict(random.sample(data_object.items(), sample_size))
        elif isinstance(data_object, pd.DataFrame):
            return data_object.sample(n=sample_size)


class FeatureEngineering_NER(FeatureEngineering):
    def __init__(self, data_args):
        self.labels_dict = data_args['task'].labels_dict
        self.data = data_args['dataset']
        self.candidates_fields = data_args['candidates_fields']

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


class FeatureEngineering_MLP(FeatureEngineering):
    def __init__(self, data_args):
        self.task = data_args['task']
        self.data = data_args['dataset']
        self.data_config = data_args['data_config']

    def create_train_test_data(self):
        # Balance samples by removing some non-entity labeled datapoints
        samples, targets, warnings = self.data.generate_spatial_samples('employer_name', self.data_config)
        pos_idx = np.where(targets == 1)[0]
        num_pos_samples = len(pos_idx)

        neg_idxs_all = np.where(targets == 0)[0]
        np.random.shuffle(neg_idxs_all)
        neg_idx = neg_idxs_all[:num_pos_samples]

        idx_to_use = np.concatenate((pos_idx, neg_idx))

        filtered_samples = samples[idx_to_use]
        filtered_targets = targets[idx_to_use]

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(filtered_samples, filtered_targets, test_size=0.3, random_state=0)
        return (X_train, X_test, y_train, y_test)

class ModelEvaluator():
    def __init__(self, training_args):
        self.labels_dict = training_args['task'].labels_dict
        self.cat_dict = training_args['task'].cat_dict
        self.training_args = training_args

    def get_Retr_Rel_Set(self, retrieved, relevant):
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

    def get_Precision_Recall(self, retrieved, relevant):
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
        model_file_or_path = self.training_args['model_file_or_path']
        self.num_labels = self.training_args['num_labels']
        gpu = self.training_args['gpu']
        if self.training_args['use_goldens']: 
            # testdata is single dataframe as data is generated using goldens csv
            logging.info("inferring BERT classifier for single df generated from goldens csv of size {}".format(testdata.shape))
            logging.info("Make sure training_args.use_goldens is set to True")
            return get_classifier_inference(model_file_or_path, testdata, self.num_labels, gpu)
        else:
            # test_data is a dictionary {'filename' : dataframe}
            results = {}
            for key in testdata:
                logging.info("inferring BERT classifier for file {}".format(key))
                results[key] = get_classifier_inference(model_file_or_path, testdata[key], self.num_labels, gpu)
            return results

    def analyze_golden_result(self, results):
        labels = [x for x in range(self.num_labels - 1)] # Not to include 'None' class since it is never in true_label
        true_labels = results['label']
        predictions = results['predicted']

        # Overall Score
        micro_precision = precision_score(true_labels, predictions, labels=labels, average="micro")
        micro_recall = recall_score(true_labels, predictions,  labels=labels, average="micro")
        micro_f1 = f1_score(true_labels, predictions, labels=labels, average="micro")
        logging.info('Micro Test R: {0:0.4f}, P: {1:0.4f}, F1: {2:0.4f}'.format(micro_recall, micro_precision, micro_f1))

        macro_precision = precision_score(true_labels, predictions, labels=labels, average="macro")
        macro_recall = recall_score(true_labels, predictions, labels=labels, average="macro")
        macro_f1 = f1_score(true_labels, predictions, labels=labels, average="macro")
        logging.info('Macro Test R: {0:0.4f}, P: {1:0.4f}, F1: {2:0.4f}'.format(macro_recall, macro_precision, macro_f1))

        # Class Wise Score
        for lab in labels:
            # micro vs macro doesn't matter in case of single-label
            precision = precision_score(true_labels, predictions, labels=[lab], average="micro")
            recall = recall_score(true_labels, predictions,  labels=[lab], average="micro")
            f1 = f1_score(true_labels, predictions, labels=[lab], average="micro")
            logging.info('Category: {0}, Test R: {1:0.4f}, P: {2:0.4f}, F1: {3:0.4f}'.format(self.cat_dict[lab], recall, precision, f1))
    
    def analyze_overall_result(self, results, goldens, candidates_fields):
        # results is a list of dataframes (one for each file) -- BERT's outputs
        person_recall = []
        person_precision = []
        org_recall = []
        org_precision = []

        final_results = {}
        final_results['person'] = {}
        final_results['org'] = {}

        for key in results:
            bert_results = results[key]
            bert_results['predicted'] = bert_results['predicted'].astype(int)
            bert_person_df = bert_results[bert_results['predicted'] == self.labels_dict['person']]
            bert_org_df = bert_results[bert_results['predicted'] == self.labels_dict['org']]
            
            retrieved_person = bert_person_df['context'].tolist()
            retrieved_org = bert_org_df['context'].tolist()
            relevant_person = goldens.loc[key, candidates_fields['person']]
            relevant_org = goldens.loc[key, candidates_fields['org']]
            
            # PERSON NAMES
            retrieved, relevant = self.get_Retr_Rel_Set(retrieved_person, relevant_person)
            precision, recall = self.get_Precision_Recall(retrieved, relevant)
            person_precision.append(precision)
            person_recall.append(recall)
            final_results['person'][key] = retrieved

            # ORG NAMES
            retrieved, relevant = self.get_Retr_Rel_Set(retrieved_org, relevant_org)
            precision, recall = self.get_Precision_Recall(retrieved, relevant)
            org_precision.append(precision)
            org_recall.append(recall)
            final_results['org'][key] = retrieved
            

        r = np.mean(person_recall)
        p = np.mean(person_precision)
        f1 = 2*p*r/(p + r)
        logging.info("For field {0}, recall: {1:0.4f}, precision: {2:0.4f}, F1: {3:0.4f} ".format('person', r, p, f1))

        r = np.mean(org_recall)
        p = np.mean(org_precision)
        f1 = 2*p*r/(p + r)
        logging.info("For field {0}, recall: {1:0.4f}, precision: {2:0.4f}, F1: {3:0.4f} ".format('org', r, p, f1))
        return final_results

    def print_scores(self, all_recall, all_precision, person_name_models, org_name_models):
        logging.info("\nPerson Name Scores")
        for model in person_name_models:
            r = np.mean(all_recall[model])
            p = np.mean(all_precision[model])
            f1 = 2*p*r/(p + r)
            logging.info("For model {0}, recall: {1:0.4f}, precision: {2:0.4f}, F1: {3:0.4f} ".format(model, r, p, f1))

        logging.info("\nOrg Name Scores")
        for model in org_name_models:
            r = np.mean(all_recall[model])
            p = np.mean(all_precision[model])
            f1 = 2*p*r/(p + r)
            logging.info("For model {0}, recall: {1:0.4f}, precision: {2:0.4f}, F1: {3:0.4f} ".format(model, r, p, f1))

    def analyze_refiner_results(self, result_file_path, goldens, candidates_fields):
        models = ['names_vontell', 'names_token_matcher']
        spacy_models = ['names_spacy', 'org_spacy']

        # add bert in below list, if precomputed results available from analyze_overall_result()
        person_name_models = ['names_vontell', 'names_token_matcher', 'names_spacy']
        org_name_models = ['org_spacy']

        all_recall = {}
        all_precision = {}

        final_results = {}
        final_results['person'] = {}
        final_results['org'] = {}

        f = open(result_file_path)
        results = json.load(f)
        for result in results:
            # one file at a time
            key = result['absolute_ibocr_path'].split('/')[-1]
            key = ".".join(key.split('.')[:-1])
            names = {}
            phrases = result['refined_phrases']
            for phrase in phrases:
                # one model/refiner field at a time
                label = phrase['label']
                if label in models:
                    word_list = ast.literal_eval(phrase["word"])
                    cleaned_word_list = [" ".join(w.split()) for w in word_list]

                elif label in spacy_models:
                    jsonstr = phrase["word"]
                    json_dict = json.loads(jsonstr)
                    entities = json_dict["entities"]
                    cleaned_word_list = [" ".join(ent_json['entity'].split()) for ent_json in entities]
                    # remove all occurences of empty string
                    cleaned_word_list = list(filter(('').__ne__, cleaned_word_list))


                names[label] = cleaned_word_list
        
            relevant_person = goldens.loc[key, candidates_fields['person']]
            relevant_org = goldens.loc[key, candidates_fields['org']]

            # PERSON NAMES
            for model in person_name_models:
                retrieved, relevant = self.get_Retr_Rel_Set(names[model], relevant_person)
                precision, recall = self.get_Precision_Recall(retrieved, relevant)
                
                if model in final_results['person']:
                    final_results['person'][model][key] = retrieved
                else:
                    final_results['person'][model] = {}

                if model in all_recall:
                    all_recall[model].append(recall)
                    all_precision[model].append(precision)
                else:
                    all_recall[model] = [recall]
                    all_precision[model] = [precision]    
            
            
            # ORG NAMES
            for model in org_name_models:
                retrieved, relevant = self.get_Retr_Rel_Set(names[model], relevant_org)
                precision, recall = self.get_Precision_Recall(retrieved, relevant)
                        
                if model in final_results['org']:
                    final_results['org'][model][key] = retrieved
                else:
                    final_results['org'][model] = {}

                if model in all_recall:
                    all_recall[model].append(recall)
                    all_precision[model].append(precision)
                else:
                    all_recall[model] = [recall]
                    all_precision[model] = [precision]    

        self.print_scores(all_recall, all_precision, person_name_models, org_name_models)
        return final_results
    
class OCRUtils:
    @staticmethod
    def get_polys_within_range(word_polys,
                                min_x=float('-inf'),
                                max_x=float('inf'),
                                min_y=float('-inf'),
                                max_y=float('inf'),
                                entirely_contained=False):
        in_range = []
        # Make this faster with binary search
        for word in word_polys:
            start_x, start_y = word['start_x'], word['start_y']
            end_x, end_y = word['end_x'], word['end_y']
            should_include = True
            if entirely_contained:
                should_include = min_x >= start_x <= max_x and min_x >= end_x <= max_x and min_y >= start_y <= max_y and min_y >= end_y <= max_y
            else:
                # If one rectangle is on left side of other
                if (min_x >= end_x or start_x >= max_x):
                    should_include = False
                # If one rectangle is above other
                elif (min_y >= end_y or start_y >= max_y):
                    should_include = False
                else:
                    should_include = True
            if should_include:
                in_range.append(word)
        return in_range

    @staticmethod
    def get_polys_in_direction(direction, word, word_polys):
        results = []
        if direction == 'above':
            results = OCRUtils.get_polys_within_range(word_polys,
                                                    min_x=word['start_x'],
                                                    max_x=word['end_x'],
                                                    max_y=word['start_y'])
        if direction == 'below':
            results = OCRUtils.get_polys_within_range(word_polys,
                                                    min_x=word['start_x'],
                                                    max_x=word['end_x'],
                                                    min_y=word['end_y'])
        if direction == 'left':
            results = OCRUtils.get_polys_within_range(word_polys,
                                                    max_x=word['start_x'],
                                                    min_y=word['start_y'],
                                                    max_y=word['end_y'])
        if direction == 'right':
            results = OCRUtils.get_polys_within_range(word_polys,
                                                    min_x=word['end_x'],
                                                    min_y=word['start_y'],
                                                    max_y=word['end_y'])
        if direction == 'above left':
            results = OCRUtils.get_polys_within_range(word_polys,
                                                    max_x=word['start_x'],
                                                    max_y=word['start_y'])
        if direction == 'above right':
            results = OCRUtils.get_polys_within_range(word_polys,
                                                    min_x=word['end_x'],
                                                    max_y=word['start_y'])
        if direction == 'below right':
            results = OCRUtils.get_polys_within_range(word_polys,
                                                    min_x=word['end_x'],
                                                    min_y=word['end_y'])
        if direction == 'below left':
            results = OCRUtils.get_polys_within_range(word_polys,
                                                    max_x=word['start_x'],
                                                    min_y=word['end_y'])
        if word in results:
            results.remove(word)
        return results

    @staticmethod
    def get_distance_between(word1, word2):
        (x1, y1, x1b,
        y1b) = word1['start_x'], word1['start_y'], word1['end_x'], word1['end_y']
        (x2, y2, x2b,
        y2b) = word2['start_x'], word2['start_y'], word2['end_x'], word2['end_y']
        left = x2b < x1
        right = x1b < x2
        bottom = y2b < y1
        top = y1b < y2
        if top and left:
            return FeatureEngineering.dist((x1, y1b), (x2b, y2))
        elif left and bottom:
            return FeatureEngineering.dist((x1, y1), (x2b, y2b))
        elif bottom and right:
            return FeatureEngineering.dist((x1b, y1), (x2, y2b))
        elif right and top:
            return FeatureEngineering.dist((x1b, y1b), (x2, y2))
        elif left:
            return x1 - x2b
        elif right:
            return x2 - x1b
        elif bottom:
            return y1 - y2b
        elif top:
            return y2 - y1b
        else:  # rectangles intersect
            return 0

    @staticmethod
    def get_tokens_that_match(text, word_polys):
        pass

class IBDOCFeaturizer:

    DIRS = [
        'left', 'above left', 'above', 'above right', 'right', 'below right',
        'below', 'below left'
    ]
    DIRS_CARDINAL = [
        'left', 'above', 'right', 'below'
    ]

    def __init__(self, ibdoc):
        self.ibdoc = ibdoc
        self.CACHED_ALL_TOKENS = None
        self.CACHED_EMBEDDINGS = None

    def get_all_tokens(self):
        if self.CACHED_ALL_TOKENS is None:
            self.CACHED_ALL_TOKENS = []
        for line in self.ibdoc.get_lines():
            for word in line:
                self.CACHED_ALL_TOKENS.append(word)
        return self.CACHED_ALL_TOKENS

    def get_bert_embedding(self, text):
        global EMBEDDING_CACHE
        text = re.sub('\d', '9', text)
        all_words = [text]
        update_bert_embeddings(all_words, EMBEDDING_CACHE)
        return EMBEDDING_CACHE.bert[text]

    def get_glove_embeddings(self):
        global EMBEDDING_CACHE
        all_words = [w['word'] for w in self.get_all_tokens()]

    def _apply_preprocessing(self, word, data_config):
        final_word = word
        for preprocess in data_config['pre_processing']:
            if type(preprocess) == str:
                if preprocess in preprocessing_rules:
                    final_word = preprocessing_rules[preprocess](final_word)
        return final_word

    def get_token_feature_vector(self, idx, data_config):

        num_directional_neighbors = data_config['max_num_tokens']
        directions = self.DIRS_CARDINAL if data_config['cardinal_only'] else self.DIRS

        pieces = []

        # First attach a feature vector for this word itself
        current_token = self.get_all_tokens()[idx]
        pieces.append(self.get_bert_embedding(self._apply_preprocessing(current_token['word'], data_config)))

        # Attach surrounding contexts
        surrounding_context = self.get_surrounding_context(
            idx, data_config)
        for direction in directions:
            for i in range(num_directional_neighbors):
                if i >= len(surrounding_context[direction]):
                    pieces.append(np.zeros((768, )))
                else:
                    word = surrounding_context[direction][i]
                    pieces.append(self.get_bert_embedding(self._apply_preprocessing(word['word'], data_config)))

        # Attach rule-based features
        # TODO
        for rule in data_config['additional_features']:
            if type(rule) == str:
                if rule in rules:
                    # print("Appending {} rule".format(rule))
                    pieces.append([rules[rule](current_token)])

        return np.concatenate(pieces)

    def get_surrounding_context(self, token_idx, data_config):
        current_token = self.get_all_tokens()[token_idx]
        context = {}
        directions = self.DIRS_CARDINAL if data_config['cardinal_only'] else self.DIRS
        for direction in directions:
            in_direction = OCRUtils.get_polys_in_direction(direction, current_token,
                                                            self.get_all_tokens())

            # Filter by allowed distance, trim if too far
            if data_config['max_token_distance'] is not None:
                filter_fn = lambda word: OCRUtils.get_distance_between(current_token, word) <= data_config['max_token_distance'] * current_token['line_height']
                in_direction = filter(filter_fn, in_direction)
            
            # Sort results by distance, trim off based on max num tokens
            result = sorted(in_direction,
                            key=lambda word: OCRUtils.get_distance_between(
                                current_token, word))[:data_config['max_num_tokens']]

        

            context[direction] = result
        return context

    def get_feature_vectors(self, data_config):

        results = []
        for idx in range(len(self.get_all_tokens())):
            results.append(
                self.get_token_feature_vector(idx, data_config))
        return np.array(results)
