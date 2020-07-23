

import string
PUNC_TABLE = str.maketrans({key: None for key in string.punctuation})

def lower_case(word):
  return word.lower()

def remove_punc(word):
  return word.translate(PUNC_TABLE)

preprocessing_rules = {
  'lower_case': lower_case,
  'remove_punc': remove_punc
}