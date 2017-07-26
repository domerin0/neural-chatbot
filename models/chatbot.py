'''
This is adapted from tensorflow translation example:
https://github.com/tensorflow/tensorflow/blob/e39d8feebb9666a331345cd8d960f5ade4652bba/tensorflow/models/rnn/translate/seq2seq_model.py

UPDATE 2017/07/20 **

revamped this model for tf 1.2 and new tf.contrib.seq2seq libraries

'''
import random

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq
from tensorflow.python.layers.core import Dense
import special_vocab as config
import util.vocabutils as vocab_utils

class ChatbotModel(object):
    def __init__(self, vocab_size, hidden_size, dropout,
                 num_layers, max_gradient_norm, batch_size, learning_rate,
                 lr_decay_factor, max_target_length,
                 max_source_length, decoder_mode=False):
        '''
        vocab_size: number of vocab tokens
        buckets: buckets of max sequence lengths
        hidden_size: dimension of hidden layers
        num_layers: number of hidden layers
        max_gradient_norm: maximum gradient magnitude
        batch_size: number of training examples fed to network at once
        learning_rate: starting learning rate of network
        lr_decay_factor: amount by which to decay learning rate
        num_samples: number of samples for sampled softmax
        decoder_mode: Whether to build backpass nodes or not
        '''
        GO_ID = config.GO_ID
        EOS_ID = config.EOS_ID
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = learning_rate
        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.source_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='source_lengths')

        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
        self.target_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name="target_lengths")

        with tf.variable_scope('embeddings') as scope:
            embeddings = tf.Variable(tf.random_uniform([vocab_size, hidden_size], -1.0, 1.0), dtype=tf.float32)
            encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs)
            targets_embedding = tf.nn.embedding_lookup(embeddings, self.decoder_targets)


        with tf.variable_scope('encoder') as scope:
            encoder_cell = rnn.LSTMCell(hidden_size)
            encoder_cell = rnn.DropoutWrapper(encoder_cell,
                                              input_keep_prob=dropout)

            encoder_cell = rnn.MultiRNNCell([encoder_cell] * num_layers)

            _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                               cell_bw=encoder_cell,
                                                               sequence_length=self.source_lengths,
                                                               inputs=encoder_inputs_embedded,
                                                               dtype=tf.float32,
                                                               time_major=False)

        with tf.variable_scope('decoder') as scope:
            decoder_cell = rnn.LSTMCell(hidden_size)
            decoder_cell = rnn.DropoutWrapper(decoder_cell,
                                              input_keep_prob=dropout)

            decoder_cell = rnn.MultiRNNCell([decoder_cell] * num_layers)

            #TODO add attention
            #seq2seq.BahdanauAttention(num_units=,memory=encoder_output)

            #decoder_cell = seq2seq.AttentionWrapper(cell=decoder_cell,
            #                                        attention_mechanism=)

        if decoder_mode:
            decoder = seq2seq.BeamSearchDecoder(embedding=embeddings,
                                                start_tokens=tf.tile([GOD_ID], [batch_size]),
                                                end_token=EOS_ID,
                                                initial_state=encoder_state[0],
                                                beam_width=2)
        else:
            helper = seq2seq.TrainingHelper(inputs=targets_embedding,
                                            sequence_length=self.target_lengths)

            decoder = seq2seq.BasicDecoder(cell=decoder_cell,
                                           helper=helper,
                                           initial_state=encoder_state[-1],
                                           output_layer=Dense(vocab_size))

        final_outputs, final_state, final_sequence_lengths =\
                            seq2seq.dynamic_decode(decoder=decoder)

        self.logits = final_outputs.rnn_output

        if not decoder_mode:
            with tf.variable_scope("loss") as scope:
                #have to pad logits, dynamic decode produces results not consistent
                #in shape with targets
                pad_size = self.max_target_length - tf.reduce_max(final_sequence_lengths)
                self.logits = tf.pad(self.logits, [[0, 0], [0,pad_size], [0, 0]])

                weights = tf.sequence_mask(lengths=final_sequence_lengths,
                                           maxlen=self.max_target_length,
                                           dtype=tf.float32,
                                           name='weights')

                x_entropy_loss = seq2seq.sequence_loss(logits=self.logits,
                                                       targets=self.decoder_targets,
                                                       weights=weights)

                self.loss = tf.reduce_mean(x_entropy_loss)

            optimizer = tf.train.AdamOptimizer()
            gradients = optimizer.compute_gradients(x_entropy_loss)
            capped_grads = [(tf.clip_by_value(grad, -max_gradient_norm, max_gradient_norm), var) for grad, var in gradients]
            self.train_op = optimizer.apply_gradients(capped_grads,
                                                      global_step=self.global_step)
            self.saver = tf.train.Saver(tf.global_variables())

    def step(self, sess, inputs,
             targets, source_lengths,
             target_lengths, test_mode=False):
        '''
        '''
        if test_mode:
            loss = sess.run([self.loss],
                    {self.encoder_inputs : inputs,
                     self.source_lengths : source_lengths,
                     self.decoder_targets : targets,
                     self.target_lengths : target_lengths})
        else:
            _, loss = sess.run([self.train_op, self.loss],
                    {self.encoder_inputs : inputs,
                     self.source_lengths : source_lengths,
                     self.decoder_targets : targets,
                     self.target_lengths : target_lengths})
        return loss

    def get_batch(self, dataset):
        '''
        Obtains batch from dataset
        Inputs: dataset - list of [input, target] sentence pairs

        Outputs:
        source_batch_major- [batch_size x max_sequence_length] inputs
        target_batch_major- [batch_size x max_sequence_length] targets
        source_seq_lengths- list of input seq lengths
        target_seq_lengths- list of target seq lengths
        '''
        source_seq_lengths = []
        target_seq_lengths =[]

        for seq in dataset:
            source_seq_lengths.append(len(seq[0]))
            target_seq_lengths.append(len(seq[1]))

        #max_target_length = max(target_seq_lengths)

        source_batch_major = np.zeros(shape=[len(dataset), self.max_source_length], dtype=np.int32)

        target_batch_major = np.zeros(shape=[len(dataset), self.max_target_length], dtype=np.int32)

        for i, seq in enumerate(dataset):
            for j, element in enumerate(seq[0]):
                source_batch_major[i, j] = element
            for j, element in enumerate(seq[1]):
                target_batch_major[i,j] = element

        return source_batch_major, target_batch_major, source_seq_lengths, target_seq_lengths
