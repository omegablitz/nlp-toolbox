
import string
PUNC_TABLE = str.maketrans({key: None for key in string.punctuation})

def is_number(token):
  word = token['word']
  try:
    float(word)
    return 1.0
  except:
    return 0.0

COMPANY_INDICATORS = set(['llc', 'co', 'ltd', 'corp', 'corporation', 'services', 'service', 'inc', 'limited', 'sons'])

def is_company_indicator(token):
  word = token['word']
  if word.lower().translate(PUNC_TABLE) in COMPANY_INDICATORS:
    return 1.0
  return 0.0

def is_date(token):
  raise NotImplementedError()

rules = {
  'is_number': is_number,
  'is_company_indicator': is_company_indicator
}