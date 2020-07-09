import csv
import json
import os
import pandas as pd
import re
import string
import logging
import numpy as np
import math

from transformers import BertModel
from transformers import BertTokenizer
import numpy as np
import torch
from torch import nn
import logging
import itertools

# Point to instabase SDK, use SDK to download
import sys
sys.path.append('/Users/ahsaasbajaj/Documents/Code/instabase/sdk/src/py')
import importlib
import instabase_sdk
importlib.reload(instabase_sdk)
from instabase_sdk import Instabase

PUNC_TABLE = str.maketrans({key: None for key in string.punctuation})

from rule_features import rules
from preprocessing import preprocessing_rules

# Import instabase
import sys
sys.path.append(
    '/Users/ahsaasbajaj/Documents/Code/instabase/distributed-tasks/celery/app-tasks/build/py'
)
from instabase.ocr.client.libs.ibocr import ParsedIBOCRBuilder

class DatasetWarning:

  def __init__(self, warning_type, message):
    self.warning_type = warning_type
    self.message = message

  def __repr__(self):
    return 'Warning[{}, {}]'.format(self.warning_type, self.message)

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

class EmbeddingCache:

  def __init__(self):
    # Model cache
    self.glove_model = None
    self.bert_model = None

    # Result cache
    self.glove = {}
    self.bert = {}


# A type for parameters in creating and training models
class ModelContext:
  """
  Args:
      field_to_capture - The field to train on from the goldens
      embedding_type - The embedding type for tokens. Can be 'bert' or 'glove'
      cardinal_only - If True, only looks above, below, left, and right of a given
                       token for surrounding context. Otherwise, also looks above + left,
                       above + right, below + left, and below + right.
      max_token_distance - The maximum distance in pixel space that one token can be from
                           another to be included within the surrounding context, in units
                           of line height from the base token.
      max_num_tokens - The maximum number of tokens to include in the context
                       from a given direction.
      additional_features - Additional features to append to each feature vector. If a string,
                            will use a built-in feature type. If a function, will pass that
                            function all tokens and the index of the current token, and will
                            expect a numpy array to be returned.
  """

  def __init__(self, **kwargs):
    self.embedding_type=kwargs.get('embedding_type', 'bert')
    self.cardinal_only=kwargs.get('cardinal_only', False)
    self.max_token_distance=kwargs.get('max_token_distance', None)
    self.max_num_tokens=kwargs.get('max_num_tokens', 5)
    self.additional_features=kwargs.get('additional_features', [])
    self.epochs=kwargs.get('epochs', 5)
    self.batch_size=kwargs.get('batch_size', 64)
    self.balance_targets=kwargs.get('balance_targets', False)
    self.pre_processing=kwargs.get('pre_processing', [])

  def __repr__(self):
    return json.dumps({
      'embedding_type': self.embedding_type,
      'cardinal_only': self.cardinal_only,
      'max_token_distance': self.max_token_distance,
      'max_num_tokens': self.max_num_tokens,
      'additional_features': str(self.additional_features),
      'epochs': self.epochs,
      'batch_size': self.batch_size,
      'balance_targets': self.balance_targets,
      'pre_processing': str(self.pre_processing)
    })

  def __str__(self):
    return repr(self)


# Cache for word embeddings
EMBEDDING_CACHE = EmbeddingCache()


class PrototypeDataset():

  def __init__(self, access_token, dataset_paths, dataset_config,
               goldens_paths, goldens_config):
    self.ib = Instabase('https://instabase.com', access_token)

    if type(dataset_paths) != list:
      dataset_paths = [dataset_paths]

    if type(goldens_paths) != list:
      goldens_paths = [goldens_paths]

    self._load_goldens(goldens_paths, goldens_config)
    self._load_ib_dataset(dataset_paths, dataset_config)

  def _row_to_mapping(self, mapping, row):
    result = {}
    for i, item in enumerate(row):
      name = mapping[i]
      result[name] = item
    return result

  def _load_goldens(self, goldens_paths, goldens_config):
    self.golden = {}
    goldens_type = goldens_config.get('file_type')
    for goldens_path in goldens_paths:
      logging.info('Loading goldens from {}'.format(goldens_path))
      if goldens_type == 'csv':
        skip_first_row = goldens_config.get('skip_first_row', False)
        with open(goldens_path) as f:
          for row in csv.reader(f):
            if skip_first_row:
              skip_first_row = False
              continue
            mapped = self._row_to_mapping(goldens_config.get('mapping'), row)
            identifier = mapped.pop(goldens_config.get('identifier'))
            self.golden[identifier] = mapped
    self.golden = pd.DataFrame.from_dict(self.golden, orient='index')

  def _load_ib_dataset(self, dataset_paths, dataset_config):
    self.dataset = {}
    for dataset_path in dataset_paths:
      logging.info('Loading dataset from {}'.format(dataset_path))
      file_result = self.ib.list_dir(dataset_path)
      files = [node['full_path'] for node in file_result['nodes']]
      for file in files:
        content = None
        identifier = file
        if dataset_config.get('file_type') == 'ibdoc':
          ibocr, err = ParsedIBOCRBuilder.load_from_ibdoc(
              self.ib.read_file(file))
          content = ibocr.as_parsed_ibocr()
        if dataset_config.get('identifier'):
          identifier = dataset_config.get('identifier')(file)
        self.dataset.update({identifier: content})

  def generate_spatial_samples(self,
                               field_to_capture,
                               model_context):
    """
    Generates samples to train a model on, based on contextual information
    surrounding a token.
    """
    
    samples = []
    targets = []
    warnings = []

    for sample_name in list(self.golden.index):
      print('-------------------------')

      if sample_name not in self.dataset:
        warnings.append(DatasetWarning('SampleNotFound', 'Golden file "{}" not found in dataset'.format(sample_name)))
        continue
          
      ibdoc = self.dataset[sample_name].get_joined_page()[0]
      featurizer = IBDOCFeaturizer(ibdoc)
      
      expected_field = self.golden.at[sample_name, field_to_capture]

      if len(expected_field.strip()) == 0:
        warnings.append(DatasetWarning('FieldEmpty', 'Golden file "{}" has no entry for "{}"'.format(sample_name, field_to_capture)))
        continue

      # First, collect a number of samples relevant for the task
      expected_tokens = [clean_token(w) for w in expected_field.split()]
      expected = ' '.join(expected_tokens)
      all_tokens = featurizer.get_all_tokens()
      found_indices = set()
      for i in range(len(all_tokens) - len(expected_tokens)):
        token_range = all_tokens[i:i+len(expected_tokens)]
        tokens_to_compare = [clean_token(w['word']) for w in token_range]
        if ' '.join(tokens_to_compare) == expected:
          for idx in range(i, i+len(expected_tokens)):
            found_indices.add(idx)
      if not found_indices:
        warnings.append(DatasetWarning('TargetNotFound', 'Not able to locate field "{}" in file "{}"'.format(field_to_capture, sample_name)))
        continue

      # Collect negative examples
      if not model_context.balance_targets:
        print("NOT BALANCING TARGETS")
        fvs = featurizer.get_feature_vectors(model_context)
        for idx in range(len(fvs)):
          samples.append(fvs[idx])
          targets.append(1 if idx in found_indices else 0)
      else:
        negative_examples = list(set(range(len(featurizer.get_all_tokens()))) - found_indices)
        selected_negative_examples = np.random.choice(negative_examples, len(found_indices), replace=False)
        fvs = []
        for idx in (list(found_indices) + list(selected_negative_examples)):
          fv = featurizer.get_token_feature_vector(idx, model_context)
          samples.append(fv)
          targets.append(1 if idx in found_indices else 0)


    return (np.array(samples), np.array(targets), warnings)

  def evaluate(self, found_mapping, comparison_fns, fields=None):
    """
        
        comparison_fns: Map<Text, (sample, field, expected, actual, result_dict): None>
        """
    results = {
        'true_positives': [],
        'false_positives': [],
        'true_negatives': [],
        'false_negatives': []
    }

    to_compare = self.golden.columns
    if fields:
      to_compare = fields

    for sample in self.golden.index:
      actual = found_mapping.get(sample, {})
      for field in to_compare:
        expected_field = self.golden.at[sample, field]
        actual_field = actual.get(field)
        if field in comparison_fns:
          comparison_fns[field](sample, field, expected_field, actual_field,
                                results)
        else:
          StrictEquality()(sample, field, expected_field, actual_field,
                           results)
    return results


def product_dict(**kwargs):
  keys = kwargs.keys()
  vals = kwargs.values()
  for instance in itertools.product(*vals):
    yield dict(zip(keys, instance))


def dist(point1, point2):
  x1, y1 = point1
  x2, y2 = point2
  return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def clean_token(token):
  return token.lower().translate(PUNC_TABLE)

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
      return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
      return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
      return dist((x1b, y1), (x2, y2b))
    elif right and top:
      return dist((x1b, y1b), (x2, y2))
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


class BERTUtils:

  @staticmethod
  def update_bert_embeddings(allsentences):
    # import pdb; pdb.set_trace()
    global EMBEDDING_CACHE
    if EMBEDDING_CACHE.bert_model:
      model, device, berttokenizer = EMBEDDING_CACHE.bert_model
    else:
      logging.info("Building BERT embeddings")
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      berttokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      model = BertModel.from_pretrained('bert-base-uncased')
      model = nn.DataParallel(model)
      model = model.to(device)
      EMBEDDING_CACHE.bert_model = (model, device, berttokenizer)
      print("FINISHED LOADING MODEL")

    for iter, sent in enumerate(allsentences):
      if sent in EMBEDDING_CACHE.bert:
        continue
      print("[WE] {}".format(sent))
      bert_tokens_sentence = berttokenizer.encode(sent,
                                                  add_special_tokens=True)
      with torch.no_grad():
        bert_embeddings = \
            model(torch.tensor([bert_tokens_sentence]).to(device))[0].squeeze(0)
        f_emb_avg = torch.mean(bert_embeddings, axis=0).cpu().numpy()
        EMBEDDING_CACHE.bert[sent] = f_emb_avg


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
    BERTUtils.update_bert_embeddings(all_words)
    return EMBEDDING_CACHE.bert[text]

  def get_glove_embeddings(self):
    global EMBEDDING_CACHE
    all_words = [w['word'] for w in self.get_all_tokens()]

  def _apply_preprocessing(self, word, model_context):
    final_word = word
    for preprocess in model_context.pre_processing:
      if type(preprocess) == str:
        if preprocess in preprocessing_rules:
          final_word = preprocessing_rules[preprocess](final_word)
    return final_word

  def get_token_feature_vector(self, idx, model_context):

    num_directional_neighbors = model_context.max_num_tokens
    directions = self.DIRS_CARDINAL if model_context.cardinal_only else self.DIRS

    pieces = []

    # First attach a feature vector for this word itself
    current_token = self.get_all_tokens()[idx]
    pieces.append(self.get_bert_embedding(self._apply_preprocessing(current_token['word'], model_context)))

    # Attach surrounding contexts
    surrounding_context = self.get_surrounding_context(
        idx, model_context)
    for direction in directions:
      for i in range(num_directional_neighbors):
        if i >= len(surrounding_context[direction]):
          pieces.append(np.zeros((768, )))
        else:
          word = surrounding_context[direction][i]
          pieces.append(self.get_bert_embedding(self._apply_preprocessing(word['word'], model_context)))

    # Attach rule-based features
    # TODO
    for rule in model_context.additional_features:
      if type(rule) == str:
        if rule in rules:
          # print("Appending {} rule".format(rule))
          pieces.append([rules[rule](current_token)])

    return np.concatenate(pieces)

  def get_surrounding_context(self, token_idx, model_context):
    current_token = self.get_all_tokens()[token_idx]
    context = {}
    directions = self.DIRS_CARDINAL if model_context.cardinal_only else self.DIRS
    for direction in directions:
      in_direction = OCRUtils.get_polys_in_direction(direction, current_token,
                                                     self.get_all_tokens())

      # Filter by allowed distance, trim if too far
      if model_context.max_token_distance is not None:
        filter_fn = lambda word: OCRUtils.get_distance_between(current_token, word) <= model_context.max_token_distance * current_token['line_height']
        in_direction = filter(filter_fn, in_direction)
      
      # Sort results by distance, trim off based on max num tokens
      result = sorted(in_direction,
                      key=lambda word: OCRUtils.get_distance_between(
                          current_token, word))[:model_context.max_num_tokens]

      

      context[direction] = result
    return context

  def get_feature_vectors(self, model_context):

    results = []
    for idx in range(len(self.get_all_tokens())):
      results.append(
          self.get_token_feature_vector(idx, model_context))
    return np.array(results)
