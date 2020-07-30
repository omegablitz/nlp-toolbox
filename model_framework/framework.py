import json
import os
import pandas as pd
import re
import string
import math
import numpy as np
import logging
import itertools
import random

# Point to instabase SDK, use SDK to download. We only need this for development, can remove this later 
# ToDo: This is fine for now, but we should make it so that this is an ENV variable (or eventually an actual pip dependency)
import sys
sys.path.append('/Users/ahsaasbajaj/Documents/Code/instabase/sdk/src/py')
import importlib
import instabase_sdk
importlib.reload(instabase_sdk)
from instabase_sdk import Instabase

PUNC_TABLE = str.maketrans({key: None for key in string.punctuation})

from bert_utils import update_bert_embeddings
from rule_features import rules
from preprocessing import preprocessing_rules

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

    @staticmethod
    def StrictEquality():

        def fn(sample_name, field, expected, actual, result_dict):
            single_result = dict(sample=sample_name,
                                field=field,
                                expected=expected,
                                actual=actual)
            if expected != actual:
                result_dict['false_negatives'].append(single_result)
            else:
                result_dict['true_positives'].append(single_result)

        return fn

    @staticmethod
    def FuzzyStringMatch(ignore_case=True,
                        reduce_whitespace=True,
                        remove_punctuation=True,
                        strip=True):

        def fn(sample_name, field, expected, actual, result_dict):
            single_result = dict(sample=sample_name,
                                field=field,
                                expected=expected,
                                actual=actual)

            if ignore_case:
                if expected is not None:
                    expected = expected.lower()
                if actual is not None:
                    actual = actual.lower()
            if remove_punctuation:
                if expected is not None:
                    expected = expected.translate(PUNC_TABLE)
                if actual is not None:
                    actual = actual.translate(PUNC_TABLE)
            if reduce_whitespace:
                if expected is not None:
                    expected = re.sub('\s+', ' ', expected)
                if actual is not None:
                    actual = re.sub('\s+', ' ', actual)
            if strip:
                if expected is not None:
                    expected = expected.strip()
                if actual is not None:
                    actual = actual.strip()

            if expected != actual:
                result_dict['false_negatives'].append(single_result)
            else:
                result_dict['true_positives'].append(single_result)

        return fn


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
                    self.golden = self.golden.set_index(goldens_config['index_field_name'])

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
                logging.info("NOT BALANCING TARGETS")
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

    @staticmethod
    def _split_train_test(data, per):
        if isinstance(data, pd.DataFrame):
            msk = np.random.rand(len(data)) < per
            train_data = data[msk]
            test_data = data[~msk]
            logging.info('Total samples {0}, training samples: {1}, Test Samples: {2}'.format(data.shape[0], train_data.shape[0], test_data.shape[0]))
            return train_data, test_data

        elif isinstance(data, dict):
            list_items = list(data.items())
            random.shuffle(list_items)

            num_train_samples = (int)(per * len(list_items))
            train_data = dict(list_items[:num_train_samples])
            test_data = dict(list_items[num_train_samples:])
            logging.info('Total samples {0}, training samples: {1}, Test Samples: {2}'.format(len(list_items), len(train_data.keys()), len(test_data.keys())))
            return train_data, test_data

    def split_train_test(self, per=0.7):
        # split dataset and as well as goldens
        # ToDo: right now, training data created from golden dataframe only, if train/test split of self.dataset required, split by same random seed

        self.golden_train, self.golden_test = self._split_train_test(self.golden, per)
        # self.data_train, self.test_data = self._split_train_test(self.dataset, per)
    
        
class ModelTrainer():
    def __init__(self, data_args,  training_args):
        self.data_args = data_args
        self.training_args = training_args


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


class Evaluation():
    @staticmethod
    def get_Retr_Rel_Set(retrieved, relevant):
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

    @staticmethod
    def get_Precision_Recall(retrieved, relevant):
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

    @staticmethod
    def print_scores(all_recall, all_precision, person_name_models, org_name_models):
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
