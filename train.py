'''
This is a chatbot based on seq2seq architecture.

This code is in part adapted from the tensorflow translation example,
'''
import math
import os
import random
import sys
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
import util.hyperparamutils as hyper_params
import util.vocabutils as vocab_utils
import util.dataprocessor as data_utils
import models.chatbot
import ConfigParser

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("lr_decay_factor", 0.99, "Learning rate decays by this much.")
flags.DEFINE_float("grad_clip", 5.0, "Clip gradients to this norm.")
flags.DEFINE_float("train_frac", 0.7, "Percentage of data to use for \
	training (rest goes into test set)")
flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
flags.DEFINE_integer("max_epoch", 6, "Maximum number of times to go over training set")
flags.DEFINE_integer("hidden_size", 100, "Size of each model layer.")
flags.DEFINE_integer("num_layers", 5, "Number of layers in the model.")
flags.DEFINE_integer("vocab_size", 40000, "Vocabulary size.")
flags.DEFINE_integer("dropout", 0.5, "Probability of hidden inputs being removed between 0 and 1.")
flags.DEFINE_string("data_dir", "data/", "Directory containing processed data.")
flags.DEFINE_string("config_file", "buckets.cfg", "path to config file contraining bucket sizes")
flags.DEFINE_string("raw_data_dir", "data/subtitles/", "Raw text data directory")
##TODO add more than one tokenizer
flags.DEFINE_string("tokenizer", "basic", "Choice of tokenizer, options are: basic, character")
flags.DEFINE_string("checkpoint_dir", "data/checkpoints/", "Checkpoint dir")
flags.DEFINE_integer("max_train_data_size", 0,
	"Limit on the size of training data (0: no limit).")
flags.DEFINE_integer("steps_per_checkpoint", 200,
	"How many training steps to do per checkpoint.")
FLAGS = tf.app.flags.FLAGS
#(10, 5), (50, 15), (100, 25),

_buckets = []
#_buckets = [(10, 10), (50, 15), (100, 20), (200, 50)]

def main():
	config = ConfigParser.ConfigParser()
	config.read(FLAGS.config_file)
	global _buckets
	_buckets = setBuckets(config.items("buckets"))
	print "Using bucket sizes:"
	print _buckets

	max_num_lines = int(config.get("max_data_sizes", "num_lines"))
	max_target_size = int(config.get("max_data_sizes", "max_target_length"))
	max_source_size = int(config.get("max_data_sizes", "max_source_length"))

	if not os.path.exists(FLAGS.checkpoint_dir):
		os.mkdir(FLAGS.checkpoint_dir)
	path = getCheckpointPath()
	print "path is {0}".format(path)
	data_processor = data_utils.DataProcessor(FLAGS.vocab_size,
		FLAGS.raw_data_dir,FLAGS.data_dir, FLAGS.train_frac, FLAGS.tokenizer,
		max_num_lines, max_target_size, max_source_size)
	data_processor.run()
	#create model
	print "Creating model with..."
	print "Number of hidden layers: {0}".format(FLAGS.num_layers)
	print "Number of units per layer: {0}".format(FLAGS.hidden_size)
	print "Dropout: {0}".format(FLAGS.dropout)
	vocab_mapper = vocab_utils.VocabMapper(FLAGS.data_dir)
	vocab_size = vocab_mapper.getVocabSize()
	print "Vocab size is: {0}".format(vocab_size)
	with tf.Session() as sess:
		writer = tf.train.SummaryWriter("/tmp/tb_logs_chatbot", sess.graph_def)
		model = createModel(sess, path, vocab_size)
		#train model and save to checkpoint
		print "Beggining training..."
		print "Maximum number of epochs to train for: {0}".format(FLAGS.max_epoch)
		print "Batch size: {0}".format(FLAGS.batch_size)
		print "Starting learning rate: {0}".format(FLAGS.learning_rate)
		print "Learning rate decay factor: {0}".format(FLAGS.lr_decay_factor)

		source_train_file_path = data_processor.data_source_train
		target_train_file_path = data_processor.data_target_train
		source_test_file_path = data_processor.data_source_test
		target_test_file_path = data_processor.data_target_test

		train_set = readData(source_train_file_path, target_train_file_path,
			FLAGS.max_train_data_size)
		test_set = readData(source_test_file_path, target_test_file_path,
			FLAGS.max_train_data_size)

		train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
		print "bucket sizes = {0}".format(train_bucket_sizes)
		train_total_size = float(sum(train_bucket_sizes))

		train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
			for i in xrange(len(train_bucket_sizes))]
		step_time, loss = 0.0, 0.0
		current_step = 0
		previous_losses = []
		while True:
			# Choose a bucket according to data distribution. We pick a random number
			# in [0, 1] and use the corresponding interval in train_buckets_scale.
			random_number_01 = np.random.random_sample()
			bucket_id = min([i for i in xrange(len(train_buckets_scale))
					   if train_buckets_scale[i] > random_number_01])

			# Get a batch and make a step.
			start_time = time.time()
			encoder_inputs, decoder_inputs, target_weights = model.get_batch(
			train_set, bucket_id)
			_, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
				target_weights, bucket_id, False)
			step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
			loss += step_loss / FLAGS.steps_per_checkpoint
			current_step += 1

			# Once in a while, we save checkpoint, print statistics, and run evals.
			if current_step % FLAGS.steps_per_checkpoint == 0:
				train_loss_summary = tf.Summary()
				str_summary_train_loss = train_loss_summary.value.add()
				str_summary_train_loss.simple_value = loss
				str_summary_train_loss.tag = "train_loss"
				writer.add_summary(train_loss_summary, current_step)
				# Print statistics for the previous epoch.
				perplexity = math.exp(loss) if loss < 300 else float('inf')
				print ("global step %d learning rate %.4f step-time %.2f perplexity "
					"%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
						 step_time, perplexity))
				# Decrease learning rate if no improvement was seen over last 3 times.
				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op)
				previous_losses.append(loss)
				# Save checkpoint and zero timer and loss.
				checkpoint_path = os.path.join(path, "chatbot.ckpt")
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				step_time, loss = 0.0, 0.0
				# Run evals on development set and print their perplexity.
				perplexity_summary = tf.Summary()
				eval_loss_summary = tf.Summary()
				for bucket_id in xrange(len(_buckets)):
					if len(test_set[bucket_id]) == 0:
						print("  eval: empty bucket %d" % (bucket_id))
						continue
					encoder_inputs, decoder_inputs, target_weights = model.get_batch(
						test_set, bucket_id)
					_, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
						target_weights, bucket_id, True)
					eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
					print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
					str_summary_ppx = perplexity_summary.value.add()
					str_summary_ppx.simple_value = eval_ppx
					str_summary_ppx.tag = "peplexity_bucket)%d" % bucket_id

					str_summary_eval_loss = eval_loss_summary.value.add()
					#need to convert from numpy.float32 to float native type
					str_summary_eval_loss.simple_value = float(eval_loss)
					str_summary_eval_loss.tag = "eval_loss_bucket)%d" % bucket_id
					writer.add_summary(perplexity_summary, current_step)
					writer.add_summary(eval_loss_summary, current_step)
				sys.stdout.flush()


def createModel(session, path, vocab_size):
	model = models.chatbot.ChatbotModel(vocab_size, _buckets,
		FLAGS.hidden_size, FLAGS.dropout, FLAGS.num_layers, FLAGS.grad_clip,
		FLAGS.batch_size, FLAGS.learning_rate, FLAGS.lr_decay_factor)
	hyper_params.saveHyperParameters(path, FLAGS)
	print path
	ckpt = tf.train.get_checkpoint_state(path)
	if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
		print "Reading model parameters from {0}".format(ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print "Created model with fresh parameters."
		session.run(tf.initialize_all_variables())
	return model

def setBuckets(raw_info):
	buckets = []
	try:
		for tu in raw_info:
			target, source = tu[1].strip().split(",")
			buckets.append((int(target), int(source)))
	except:
		print "Erorr in config file formatting..."
	return buckets


def readData(source_path, target_path, max_size=None):
	'''
	This method directly from tensorflow translation example
	'''
	data_set = [[] for _ in _buckets]
	with tf.gfile.GFile(source_path, mode="r") as source_file:
		with tf.gfile.GFile(target_path, mode="r") as target_file:
			source, target = source_file.readline(), target_file.readline()
			counter = 0
			while source and target and (not max_size or counter < max_size):
				counter += 1
				if counter % 100000 == 0:
					print("  reading data line %d" % counter)
					sys.stdout.flush()
				source_ids = [int(x) for x in source.split()]
				target_ids = [int(x) for x in target.split()]
				target_ids.append(vocab_utils.EOS_ID)
				for bucket_id, (source_size, target_size) in enumerate(_buckets):
					if len(source_ids) < source_size and len(target_ids) < target_size:
						data_set[bucket_id].append([source_ids, target_ids])
						break
				source, target = source_file.readline(), target_file.readline()
	return data_set

def getCheckpointPath():
	'''
	Check if new hyper params match with old ones
	if not, then create a new model in a new Directory
	Returns:
	path to checkpoint directory
	'''
	old_path = os.path.join(FLAGS.checkpoint_dir, "hyperparams.p")
	if os.path.exists(old_path):
		params = hyper_params.restoreHyperParams(FLAGS.checkpoint_dir)
		ok = \
		params["num_layers"] == FLAGS.num_layers and \
		params["hidden_size"] == FLAGS.hidden_size and \
		params["dropout"] == FLAGS.dropout
		if ok:
			return FLAGS.checkpoint_dir
		else:
			infostring = "hiddensize_{0}_dropout_{1}_numlayers_{2}".format(FLAGS.hidden_size,
			FLAGS.dropout, FLAGS.num_layers)
			if not os.path.exists("data/checkpoints/"):
				os.mkdirs("data/checkpoints/",)
			path = os.path.join("data/checkpoints/", str(int(time.time())) + infostring)
			if not os.path.exists(path):
				os.makedirs(path)
			print "hyper parameters changed, training new model at {0}".format(path)
			return path
	else:
		return FLAGS.checkpoint_dir

if __name__ == '__main__':
	main()
