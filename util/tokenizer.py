'''
tokenizer utility functions,
adapted from:
/tensorflow/models/rnn/translate/data_utils.py
'''
import re

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def basic_tokenizer(sequence):
  """Very basic tokenizer: split the sequence into a list of tokens."""
  words = []
  for space_separated_fragment in sequence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w.lower() for w in words if w]

def character_tokenizer(sequence):
    '''
    Very simple tokenizer, for character level models.

    Inputs:
    sequence: a string

    Returns:
    list of characters
    '''
    return list(sequence)
