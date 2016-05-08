'''
Process the raw text data to create the following:

1. vocabulary file
2. source_train_file, target_train_file (index mapped train set files)
3. source_test_file, target_test_file (index mapped test set files)

TODO:
Some very minor parallelization takes place where train and test sets are
created in parallel. A much better parallelization can be achieved. It takes too
much time to process the data currently.

Clean up the code duplication
'''


import os
import util.tokenizer
import util.vocabutils as vocab_utils
from tensorflow.python.platform import gfile
from random import shuffle
from multiprocessing import Process, Lock
import time
from math import floor

class DataProcessor(object):
    def __init__(self, max_vocab_size, source_data_path,
    processed_data_path, train_frac, tokenizer_str,
    num_lines=4, max_target_length=50, max_source_length=200, is_discrete=False,
    extra_discrete_data=""):
        '''
        Inputs:
        max_vocab_size: max size of vocab allowed
        source_data_path: path to raw data files to be processed
        processed_data_path: path to processed data directory (usually just data/)
        train_frac: fraction of data to use for training
        tokenizer_str: string, type of tokenizer to use
        num_lines: max number of lines for conversational history
        max_target_length: max length of target sentence
        max_source_length: max length of source sentence
        is_discrete: boolean indicating whether the lines are continuous or discrete pairs
        '''
        self.is_discrete = is_discrete
        self.MAX_SOURCE_TOKEN_LENGTH = max_source_length
        self.MAX_TARGET_TOKEN_LENGTH = max_target_length
        self.NUM_LINES = num_lines
        self.tokenizer = util.tokenizer.basic_tokenizer
        assert train_frac > 0.0 and train_frac <= 1.0, "Train frac not between 0 and 1..."
        self.train_frac = train_frac
        self.max_vocab_size = max_vocab_size
        self.source_data_path = source_data_path
        self.processed_data_path = processed_data_path
        self.extra_discrete_data = extra_discrete_data
        train_path = os.path.join(processed_data_path, "train/")
        test_path = os.path.join(processed_data_path, "test/")

        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        self.data_source_train = os.path.join(train_path,
            "data_source_train.txt")
        self.data_target_train = os.path.join(train_path,
            "data_target_train.txt")

        self.data_source_test = os.path.join(test_path,
            "data_source_test.txt")
        self.data_target_test = os.path.join(test_path,
            "data_target_test.txt")

        print "Checking to see what data processor needs to do..."
        vocab_path = os.path.join(processed_data_path, "vocab.txt")
        self.vocab_exists = gfile.Exists(vocab_path)

        self.data_files_exist = self.vocab_exists and \
            gfile.Exists(self.data_source_train) and \
            gfile.Exists(self.data_target_train) and \
            gfile.Exists(self.data_source_test) and \
            gfile.Exists(self.data_target_test)

    def run(self):
        if not self.data_files_exist:
            print "Obtaining raw text conversation files..."
            text_files = self.getRawFileList(self.source_data_path)
            if not self.extra_discrete_data:
                extra_files = self.getRawFileList(self.extra_discrete_data)
            else:
                extra_files = []
            # randomly shuffle order of files
            shuffle(text_files)
            num_train_files = int(self.train_frac * len(text_files))

        #create vocab file
        if not self.vocab_exists:
            vocab_builder = vocab_utils.VocabBuilder(self.max_vocab_size, self.processed_data_path)
            print "Building vocab..."
            #loop through continuous/discrete data
            for text_file in text_files:
                with open(text_file, "r+") as f:
                    vocab_builder.growVocab(f.read())
            #loopthrough extra discrete data
            for text_file in extra_files:
                with open(text_file, "r+") as f:
                    vocab_builder.growVocab(f.read())
            print "Creating vocab file..."
            vocab_builder.createVocabFile()

        if not self.data_files_exist:
            self.vocab_mapper = vocab_utils.VocabMapper(self.processed_data_path)
            #create source and target token id files
            processes = []
            print "Creating token id data source and target train files..."

            if len(text_files) == 1:
                num_train_files = 1
                text_files = self.splitSingle2Many(text_files[0], self.train_frac)
            if len(extra_files) == 1:
                num_extra_files = 1
                extra_files = self.splitSingle2Many(extra_files[0], self.train_frac)
            else:
                num_extra_files = len(extra_files)

            p1 = Process(target=self.loopParseTextFiles, args=([text_files[:num_train_files]], True, self.is_discrete))
            p1.start()
            processes.append(p1)
            print "Creating token id data source and target test files..."
            print "This is going to take a while..."
            p2 = Process(target=self.loopParseTextFiles, args=([text_files[num_train_files:]], False, self.is_discrete))
            p2.start()
            processes.append(p2)

            for p in processes:
                if p.is_alive():
                    p.join()

            if len(extra_files) > 0:
                p2 = Process(target=self.loopParseTextFiles, args=([extra_files[num_extra_files:]], False, True))
                p2.start()
                processes.append(p2)
                p1 = Process(target=self.loopParseTextFiles, args=([extra_files[:num_extra_files]], True, True))
                p1.start()
                processes.append(p1)

            for p in processes:
                if p.is_alive():
                    p.join()
            print "Done data pre-processing..."

    def loopParseTextFiles(self, text_files, is_train, is_discrete):
        for text_file in text_files[0]:
            if is_discrete:
                self.parseDiscreteTextFile(text_file, is_train)
            else:
                self.parseTextFile(text_file, is_train)

    def parseDiscreteTextFile(self, text_file, is_train):
            with open(text_file, "r+") as f:
                sentences = f.read().split("\n")
                #make sure even number of senteces to pair off
                if len(sentences) % 2 != 0:
                    del sentences[-1]
                for i in range(0, len(sentences), 2):
                    source_sentences = sentences[i].strip().lower()
                    target_sentences = sentences[i+1].strip().lower()
                    #Tokenize sentences
                    source_sentences = self.tokenizer(source_sentences)
                    target_sentences = self.tokenizer(target_sentences)

                    #Convert tokens to id string, reverse source inputs
                    source_sentences = list(reversed(self.vocab_mapper.tokens2Indices(source_sentences)))
                    target_sentences = self.vocab_mapper.tokens2Indices(target_sentences)
                    #remove outliers (really long sentences) from data
                    if len(source_sentences) >= self.MAX_SOURCE_TOKEN_LENGTH or \
                        len(target_sentences) >= self.MAX_TARGET_TOKEN_LENGTH:
                        continue
                    source_sentences = " ".join([str(x) for x in source_sentences])
                    target_sentences = " ".join([str(x) for x in target_sentences])

                    data_source = self.data_source_train
                    data_target = self.data_target_train
                    if not is_train:
                        data_source = self.data_source_test
                        data_target = self.data_target_test
                    with open(data_source, "a+") as f2:
                        f2.write(source_sentences + "\n")
                    with open(data_target, "a+") as f2:
                        f2.write(target_sentences + "\n")

    def splitSingle2Many(self, text_file, train_frac):
        '''
        Split a single data file into many files
        (to work into processing pipeline)
        '''
        temp = "temp/"
        if not gfile.Exists(temp):
            os.mkdir(temp)
        with open(text_file, 'r') as f:
            sentences = f.read().split('\n')
            num_train = int(floor(train_frac * len(sentences)))
            if num_train %2 != 0:
                num_train += 1
            num_test = len(sentences) - num_train
            print "num train {0}, num test {1}".format(num_train, num_test)
            train_file_name = "{0}{1}train.txt".format(temp,int(time.time()))
            test_file_name = "{0}{1}test.txt".format(temp,int(time.time()))
            with open(train_file_name, "w+") as f2:
                f2.write("\n".join(sentences[:num_train]))
            with open(test_file_name, "w+") as f2:
                f2.write("\n".join(sentences[num_train:]))
            return [train_file_name, test_file_name]

    def parseTextFile(self, text_file, is_train):
        with open(text_file, "r+") as f:
            line_buffer = []
            for line in f:
                if len(line_buffer) > self.NUM_LINES:
                    self.findSentencePairs(line_buffer, is_train)
                    line_buffer.pop(0)
                line_buffer.append(line)

    def getRawFileList(self, path):
        text_files = []
        for f in os.listdir(path):
            if not f.endswith("~"):
                text_files.append(os.path.join(path, f))
        return text_files


    def findSentencePairs(self, line_buffer, is_train):
        assert len(line_buffer) == self.NUM_LINES+1, "Num lines: {0}, length of line buffer: {1}".format(self.NUM_LINES, len(line_buffer))
        if len(line_buffer) > 0:
            for i in range(1, len(line_buffer)):
                source_sentences = " ".join(line_buffer[:i])
                source_sentences = source_sentences.strip()
                target_sentences = line_buffer[i].strip()
                #Tokenize sentences
                source_sentences = self.tokenizer(source_sentences)
                target_sentences = self.tokenizer(target_sentences)

                #Convert tokens to id string, reverse source inputs
                source_sentences = list(reversed(self.vocab_mapper.tokens2Indices(source_sentences)))
                target_sentences = self.vocab_mapper.tokens2Indices(target_sentences)
                #remove outliers (really long sentences) from data
                if len(source_sentences) >= self.MAX_SOURCE_TOKEN_LENGTH or \
                    len(target_sentences) >= self.MAX_TARGET_TOKEN_LENGTH:
                    print "skipped {0} and {1}".format(len(source_sentences), len(target_sentences))
                    print source_sentences
                    print target_sentences
                    continue
                source_sentences = " ".join([str(x) for x in source_sentences])
                target_sentences = " ".join([str(x) for x in target_sentences])

                data_source = self.data_source_train
                data_target = self.data_target_train
                if not is_train:
                    data_source = self.data_source_test
                    data_target = self.data_target_test

                with open(data_source, "a+") as f2:
                    f2.write(source_sentences + "\n")
                with open(data_target, "a+") as f2:
                    f2.write(target_sentences + "\n")
