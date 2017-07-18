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

_buckets = []
convo_hist_limit = 1
max_source_length = 0
max_target_length = 0
#_buckets = [(10, 10), (50, 15), (100, 20), (200, 50)]

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/1461979205hiddensize_100_dropout_0.5_numlayers_1', 'Directory to store/restore checkpoints')
flags.DEFINE_string('data_dir', "data/", "Data storage directory")
flags.DEFINE_string('static_data', '', '(path to static data) Adds fuzzy matching layer on top of chatbot for better static responses')
flags.DEFINE_integer('static_temp', 60, 'number between 0 and 100. The lower the number the less likely static responses will come up')
#flags.DEFINE_string('text', 'Hello World!', 'Text to sample with.')


#Read in static data to fuzzy matcher.
#Assumes static_data has text files with discrete (source, target) pairs
#Sources are on odd lines n_i, targets are on even lines n_{i+1}
static_sources = []
static_targets = []
if FLAGS.static_data:
	if os.path.exists(FLAGS.static_data):
		try:
			from fuzzywuzzy import fuzz
			from fuzzywuzzy import process
			onlyfiles = [f for f in listdir(FLAGS.static_data) if isfile(join(FLAGS.static_data, f))]
			for f in onlyfiles:
				with open(os.path.join(FLAGS.static_data, f), 'r') as f2:
					file_lines = f2.readlines()
					for i in range(0, len(file_lines) - 1, 2):
						static_sources.append(file_lines[i].lower().replace('\n', ''))
						static_targets.append(file_lines[i+1].lower().replace('\n', ''))
		except ImportError:
			print("Package fuzzywuzzy not found")
			print("Running sampling without fuzzy matching...")
	else:
		print("Fuzzy matching data not found... double check static_data path..")
		print("Not using fuzzy matching... Reverting to normal sampling")

def main():
	with tf.Session() as sess:
		model = load_model(sess, FLAGS.checkpoint_dir)
		print(_buckets)
		model.batch_size = 1
		vocab = vocab_utils.VocabMapper(FLAGS.data_dir)
		sys.stdout.write(">")
		sys.stdout.flush()
		sentence = sys.stdin.readline().lower()
		conversation_history = [sentence]
		while sentence:

			use_static_match = False
			if len(static_sources) > 0:
				#static_match = process.extractOne(sentence, static_sources)
				#Check is static match is close enough to original input
				best_ratio = 0
				static_match = ""
				for s in static_sources:
					score = fuzz.partial_ratio(sentence, s)
					if score > best_ratio:
						static_match = s
						best_ratio = score
				if best_ratio > FLAGS.static_temp:
					use_static_match = True
					#Find corresponding target in static list, bypass neural net output
					convo_output = static_targets[static_sources.index(static_match)]

			if not use_static_match:
				token_ids = list(reversed(vocab.tokens2Indices(" ".join(conversation_history))))
				#token_ids = list(reversed(vocab.tokens2Indices(sentence)))
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

			conversation_history.append(convo_output)
			print(convo_output)
			sys.stdout.write(">")
			sys.stdout.flush()
			sentence = sys.stdin.readline().lower()
			conversation_history.append(sentence)
			conversation_history = conversation_history[-convo_hist_limit:]

def load_model(session, path):
	global _buckets
	global max_source_length
	global max_target_length
	global convo_hist_limit
	params = hyper_params.restoreHyperParams(path)
	buckets = []
	num_buckets = params["num_buckets"]
	max_source_length = params["max_source_length"]
	max_target_length = params["max_target_length"]
	convo_hist_limit = params["conversation_history"]
	for i in range(num_buckets):
		buckets.append((params["bucket_{0}_target".format(i)],
			params["bucket_{0}_target".format(i)]))
		_buckets = buckets
	model = models.chatbot.ChatbotModel(params["vocab_size"], _buckets,
		params["hidden_size"], 1.0, params["num_layers"], params["grad_clip"],
		1, params["learning_rate"], params["lr_decay_factor"], 512, True)
	ckpt = tf.train.get_checkpoint_state(path)
	if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
		print("Reading model parameters from {0}".format(ckpt.model_checkpoint_path))
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Double check you got the checkpoint_dir right...")
		print("Model not found...")
		model = None
	return model


if __name__=="__main__":
	main()
