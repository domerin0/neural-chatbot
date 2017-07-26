'''
This is a chatbot based on seq2seq architecture.

'''
import math
import os
import random
import sys
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import gfile
import util.hyperparamutils as hyper_params
import util.vocabutils as vocab_utils
import util.dataprocessor as data_utils
import models.chatbot as chatbot

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("lr_decay_factor", 0.97, "Learning rate decays by this much.")
flags.DEFINE_float("grad_clip", 5.0, "Clip gradients to this norm.")
flags.DEFINE_float("train_frac", 0.7, "Percentage of data to use for \
	training (rest goes into test set)")
flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
flags.DEFINE_integer("max_epoch", 6, "Maximum number of times to go over training set")
flags.DEFINE_integer("hidden_size", 200, "Size of each model layer.")
flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
flags.DEFINE_integer("vocab_size", 40000, "Max vocabulary size.")
flags.DEFINE_integer("dropout", 0.8, "Probability of hidden inputs being removed between 0 and 1.")
flags.DEFINE_string("data_dir", "data/", "Directory containing processed data.")
flags.DEFINE_string("raw_data_dir", "data/cornell_lines/", "Raw text data directory")
##TODO add more than one tokenizer
flags.DEFINE_string("tokenizer", "basic", "Choice of tokenizer, options are: basic (for now)")
flags.DEFINE_string("checkpoint_dir", "data/checkpoints/", "Checkpoint dir")
flags.DEFINE_integer("max_train_data_size", 0,
	"Limit on the size of training data (0: no limit).")
flags.DEFINE_integer("steps_per_checkpoint", 200,
	"How many training steps to do per checkpoint.")
flags.DEFINE_integer("max_target_length", 50, "max length of target response")
flags.DEFINE_integer("max_source_length", 75, "max length of source input")
flags.DEFINE_integer("convo_limits", 1, "how far back the conversation memory should be")
FLAGS = tf.app.flags.FLAGS

def main():
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.mkdir(FLAGS.checkpoint_dir)
	path = get_checkpoint_path()
	print("path is {0}".format(path))
	data_processor = data_utils.DataProcessor(FLAGS.vocab_size,
		FLAGS.raw_data_dir,FLAGS.data_dir, FLAGS.train_frac, FLAGS.tokenizer,
		FLAGS.convo_limits, FLAGS.max_target_length, FLAGS.max_source_length)
	data_processor.run()
	#create model
	print("Creating model with...")
	print("Number of hidden layers: {0}".format(FLAGS.num_layers))
	print("Number of units per layer: {0}".format(FLAGS.hidden_size))
	print("Dropout: {0}".format(FLAGS.dropout))
	vocab_mapper = vocab_utils.VocabMapper(FLAGS.data_dir)
	vocab_size = vocab_mapper.get_vocab_size()
	print("Vocab size is: {0}".format(vocab_size))
	FLAGS.vocab_size = vocab_size

	last_test_loss = float('inf')
	with tf.Session() as sess:
		model = create_model(sess, path, vocab_size)
		#train model and save to checkpoint
		print("Beggining training...")
		print("Maximum number of epochs to train for: {0}".format(FLAGS.max_epoch))
		print("Batch size: {0}".format(FLAGS.batch_size))
		print("Starting learning rate: {0}".format(FLAGS.learning_rate))
		print("Learning rate decay factor: {0}".format(FLAGS.lr_decay_factor))

		source_train_file_path = data_processor.data_source_train
		target_train_file_path = data_processor.data_target_train
		source_test_file_path = data_processor.data_source_test
		target_test_file_path = data_processor.data_target_test
		print(source_train_file_path)
		print(target_train_file_path)

		train_set = read_data(source_train_file_path, target_train_file_path,
			FLAGS.max_train_data_size)
		random.shuffle(train_set)
		test_set = read_data(source_test_file_path, target_test_file_path,
			FLAGS.max_train_data_size)
		random.shuffle(test_set)

		step_time, train_loss = 0.0, 0.0
		current_step = 0
		previous_losses = []
		num_batches = len(train_set) / FLAGS.batch_size
		batch_pointer = 0

		while True:
			# Get a batch and make a step.
			start_time = time.time()
			start_index = int(batch_pointer * FLAGS.batch_size)
			end_index = int(start_index + FLAGS.batch_size)
			inputs, targets, input_lengths, target_lengths =\
			 	model.get_batch(train_set[start_index : end_index])
			step_loss = model.step(sess, inputs, targets,
									  input_lengths, target_lengths)
			batch_pointer = (batch_pointer + 1) % num_batches
			step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
			train_loss += step_loss / FLAGS.steps_per_checkpoint
			current_step += 1

			# Once in a while, we save checkpoint, show statistics, and run tests.
			if current_step % FLAGS.steps_per_checkpoint == 0:
				# show statistics for the previous epoch.
				print("Step {0} learning rate {1} step-time {2} training loss {3}"\
				.format(model.global_step.eval(), round(model.learning_rate,4),
						 round(step_time, 4), round(train_loss,4)))
				# Decrease learning rate if no improvement was seen over last 3 times.
				#if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
				#	sess.run(model.learning_rate_decay_op)
				previous_losses.append(train_loss)

				# Run tests on test set and show their perplexity.
				test_losses = []
				num_test_batches = int(len(test_set) / FLAGS.batch_size)
				for test_pointer in range(0, num_test_batches):
					start_index = test_pointer * FLAGS.batch_size
					inputs, targets, input_lengths, target_lengths =\
				 	model.get_batch(test_set[start_index : start_index + FLAGS.batch_size])
					test_loss = model.step(sess,
										   inputs,
										   targets,
										   input_lengths,
										   target_lengths,
										   test_mode=True)
					test_losses.append(test_loss)

				test_loss = float(np.mean(test_losses))

				print(" step: {0} test loss: {1}".format(
					model.global_step.eval(),
					round(test_loss,4)))
				# Save checkpoint and zero timer and loss.
				if test_loss < last_test_loss:
					checkpoint_path = os.path.join(path, "chatbot")
					model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				last_test_loss = test_loss
				step_time, train_loss = 0.0, 0.0

				sys.stdout.flush()


def create_model(session, path, vocab_size):
	model = chatbot.ChatbotModel(vocab_size=vocab_size,
								 hidden_size=FLAGS.hidden_size,
								 dropout=FLAGS.dropout,
								 num_layers=FLAGS.num_layers,
								 max_gradient_norm=FLAGS.grad_clip,
								 batch_size=FLAGS.batch_size,
								 learning_rate=FLAGS.learning_rate,
								 max_target_length = FLAGS.max_target_length,
					             max_source_length = FLAGS.max_source_length,
								 lr_decay_factor=FLAGS.lr_decay_factor,
								 decoder_mode=False)
	hyper_params.save_hyper_params(path, FLAGS)
	ckpt_path = tf.train.latest_checkpoint(path)
	if ckpt_path:
		print("Reading model parameters from {0}".format(ckpt_path))
		model.saver.restore(session,ckpt_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.global_variables_initializer())
	return model

def read_data(source_path, target_path, max_size=None):
	'''
	This method directly from tensorflow translation example
	'''
	data_set = []
	with tf.gfile.GFile(source_path, mode="rb") as source_file:
		with tf.gfile.GFile(target_path, mode="rb") as target_file:
			source, target = source_file.readline(), target_file.readline()
			counter = 0
			while source and target and (not max_size or counter < max_size):
				counter += 1
				if counter % 100000 == 0:
					print("Reading data line {0}".format(counter))
					sys.stdout.flush()
				source_ids = [int(x) for x in source.split()]
				target_ids = [vocab_utils.GO_ID]
				target_ids.extend([int(x) for x in target.split()])
				target_ids.append(vocab_utils.EOS_ID)
				data_set.append([source_ids, target_ids])
				source, target = source_file.readline(), target_file.readline()
	return data_set

def get_checkpoint_path():
	'''
	Check if new hyper params match with old ones
	if not, then create a new model in a new Directory
	Returns:
	path to checkpoint directory
	'''
	#check if model exists with params
	dir_name = "numlayers_{0}_hsize_{1}_vsize_{2}_max_tlength_{3}_max_slength_{4}".format(FLAGS.num_layers,
											FLAGS.hidden_size,
											FLAGS.vocab_size,
											FLAGS.max_target_length,
											FLAGS.max_source_length)
	checkpoint_path = os.path.join(FLAGS.checkpoint_dir, dir_name)
	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)
	return checkpoint_path

if __name__ == '__main__':
	main()
