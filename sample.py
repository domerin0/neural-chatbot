'''
Code in this file is for sampling use of chatbot
'''


import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import sys
import os
import nltk
from six.moves import xrange
import models.chatbot
import util.hyperparamutils as hyper_params
import util.vocabutils as vocab_utils
from os import listdir
from os.path import isfile, join

convo_hist_limit = 1
max_source_length = 0
max_target_length = 0

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/1461979205hiddensize_100_dropout_0.5_numlayers_1', 'Directory to store/restore checkpoints')
flags.DEFINE_string('data_dir', "data/", "Data storage directory")

def main():
	with tf.Session() as sess:
		model = load_model(sess, FLAGS.checkpoint_dir)
		model.batch_size = 1
		model.dropout = 1
		vocab = vocab_utils.VocabMapper(FLAGS.data_dir)
		sys.stdout.write(">")
		sys.stdout.flush()
		sentence = sys.stdin.readline().lower()
		conversation_history = [sentence]
		while sentence:

			token_ids = list(reversed(vocab.token_2_indices(" ".join(conversation_history))))
			#token_ids = list(reversed(vocab.token_2_indices(sentence)))
			#token_ids.insert(0, vocab_utils.GO_ID)
			#token_ids.append(vocab_utils.EOS_ID)
			output_logits = model.sample(sess, [token_ids])

			outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

			if vocab_utils.EOS_ID in outputs:
				outputs = outputs[:outputs.index(vocab_utils.EOS_ID)]

			convo_output =  " ".join(vocab.indices_2_tokens(outputs))

			conversation_history.append(convo_output)
			print(convo_output)
			sys.stdout.write(">")
			sys.stdout.flush()
			sentence = sys.stdin.readline().lower()
			conversation_history.append(sentence)
			conversation_history = conversation_history[-convo_hist_limit:]

def load_model(session, path):
	global max_source_length
	global max_target_length
	params = hyper_params.restore_hyper_params(path)
	buckets = []
	max_source_length = params["max_source_length"]
	max_target_length = params["max_target_length"]
	convo_hist_limit = params["convo_limits"]

	model = models.chatbot.ChatbotModel(vocab_size=params["vocab_size"],
								 		hidden_size=params["hidden_size"],
								 		dropout=1.0,
								 		num_layers=params["num_layers"],
								 		max_gradient_norm=5,
								 		batch_size=1,
								 		learning_rate=0.0,
										lr_decay_factor=0.0,
								 		max_target_length = max_target_length,
					             		max_source_length = max_source_length,
								 		decoder_mode=True)
	ckpt_path = tf.train.latest_checkpoint(path)
	if ckpt_path:
		print("Reading model parameters from {0}".format(ckpt_path))
		model.saver.restore(session,ckpt_path)
	else:
		raise Exception("No model found at {0}".format(path))
	return model


if __name__=="__main__":
	main()
