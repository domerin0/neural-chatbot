'''
Code in this file is for sampling use of chatbot
'''


import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell, seq2seq
from tensorflow.python.platform import gfile
import numpy as np
import sys
import os
import nltk
from six.moves import xrange
import models.chatbot
import util.hyperparamutils as hyper_params
import util.vocabutils as vocab_utils
_buckets = [(10, 10), (50, 15), (100, 20), (200, 50)]

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/1461170194hiddensize_200_dropout_0.8_numlayers_4', 'Directory to store/restore checkpoints')
flags.DEFINE_string('data_dir', "data/", "Data storage directory")
flags.DEFINE_string('text', 'Hello World!', 'Text to sample with.')

def main():
	with tf.Session() as sess:
		model = loadModel(sess, FLAGS.checkpoint_dir)
		model.batch_size = 1
		vocab = vocab_utils.VocabMapper(FLAGS.data_dir)
		sys.stdout.write(">")
		sys.stdout.flush()
		sentence = sys.stdin.readline()
		conversation_history = [sentence]
		while sentence:
			#token_ids = list(reversed(vocab.tokens2Indices(" ".join(conversation_history))))
			token_ids = list(reversed(vocab.tokens2Indices(sentence)))
			bucket_id = min([b for b in xrange(len(_buckets))
				if _buckets[b][0] > len(token_ids)])

			encoder_inputs, decoder_inputs, target_weights = model.get_batch(
			{bucket_id: [(token_ids, [])]}, bucket_id)

			_, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
				target_weights, bucket_id, True)

			#TODO implement beam search
			outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

			if vocab_utils.EOS_ID in outputs:
				outputs = outputs[:outputs.index(vocab_utils.EOS_ID)]

			convo_output =  " ".join(vocab.indices2Tokens(outputs))

			#conversation_history.append(convo_output)
			#if len(conversation_history) >= 1:
			#	conversation_history.pop(0)
			print convo_output
			sys.stdout.write(">")
			sys.stdout.flush()
			sentence = sys.stdin.readline()


def loadModel(session, path):
	params = hyper_params.restoreHyperParams(path)
	model = models.chatbot.ChatbotModel(params["vocab_size"], _buckets,
		params["hidden_size"], 1.0, params["num_layers"], params["grad_clip"],
		1, params["learning_rate"], params["lr_decay_factor"], 512, True)
	ckpt = tf.train.get_checkpoint_state(path)
	if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
		print "Reading model parameters from {0}".format(ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print "Double check you got the checkpoint_dir right..."
		print "Model not found..."
		model = None
	return model


if __name__=="__main__":
	main()
