import pickle
from tensorflow.python.platform import gfile
import util.tokenizer
import re
import os

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

class VocabBuilder(object):
    def __init__(self,max_vocab_size, data_path, tokenizer = None, normalize_digits = True):
        '''
        This class enables dynamic building of vocabulary file
        '''
        if tokenizer is None:
            self.tokenizer = util.tokenizer.basic_tokenizer
        else:
            self.tokenizer = tokenizer
        self.vocab = {}
        self.max_vocab_size = max_vocab_size
        self.data_path = data_path

    def growVocab(self, text, normalize_digits = True):
        tokens = self.tokenizer(text)
        for w in tokens:
            word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
            if word in self.vocab:
                self.vocab[word] += 1
            else:
                self.vocab[word] = 1

    def createVocabFile(self):
        vocab_list = _START_VOCAB + sorted(self.vocab, key=self.vocab.get, reverse=True)
        if len(vocab_list) > self.max_vocab_size:
            vocab_list = vocab_list[:self.max_vocab_size]
        vocab_path = os.path.join(self.data_path, "vocab.txt")
        with gfile.GFile(vocab_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")


class VocabMapper(object):
    def __init__(self, data_path, tokenizer = None):
        if tokenizer is None:
            self.tokenizer = util.tokenizer.basic_tokenizer
        else:
            self.tokenizer = tokenizer
        vocab_path = os.path.join(data_path, "vocab.txt")
        if gfile.Exists(vocab_path):
            rev_vocab = []
            with gfile.GFile(vocab_path, mode = "rb") as f:
                rev_vocab.extend(f.readlines())
            rev_vocab = [line.strip() for line in rev_vocab]
            vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
            self.vocab = vocab
            self.rev_vocab = rev_vocab
        else:
            raise ValueError("Vocab file not found!")

    def getVocabSize(self):
        return len(self.rev_vocab)

    def tokens2Indices(self, text):
        '''
        Inputs
        text: list of tokens (or a string)

        Returns:
        a list of ints representing token indices
        '''
        if type(text) == type("string"):
            text = self.tokenizer(text)
        indices = []
        for token in text:
            if token in self.vocab:
                indices.append(self.vocab[token])
            else:
                indices.append(UNK_ID)
        return indices

    def indices2Tokens(self, indices):
        '''
        Inputs
        indices: a list of ints representing token indices

        Returns:
        tokens: a list of tokens
        '''
        tokens = []
        for index in indices:
            tokens.append(self.rev_vocab[index])
        return tokens
