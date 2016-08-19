'''
This program plots the lengths of source input and target pairs.

The intention is for one to use this to help determine bucket sizes.

Maybe in the future I will implement a clustering algorithm to autonomously find
bucket sizes
'''



import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import nltk
import util.dataprocessor as data_utils
import sys
import numpy as np
import os
import tensorflow as tf
import ConfigParser

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "data/", "data directory")
flags.DEFINE_boolean("plot_histograms", True, "Plot histograms of sequence lengths")
flags.DEFINE_boolean("plot_scatter", True, "Make scatter plot of target vs. source length")

flags.DEFINE_integer("vocab_size", 40000, "Vocabulary size.")
flags.DEFINE_string("raw_data_dir", "data/subtitles/", "Raw text data directory")
flags.DEFINE_float("train_frac", 0.7, "Percentage of data to use for \
	training (rest goes into test set)")
flags.DEFINE_string("tokenizer", "basic", "Choice of tokenizer, options are: basic (for now)")

flags.DEFINE_string("config_file", "buckets.cfg", "path to config file contraining bucket sizes")

num_bins = 50

def main():
	dirs = [os.path.join(FLAGS.data_dir, "train/"),
		os.path.join(FLAGS.data_dir, "test/")]
	if not (os.path.exists(dirs[0]) and os.path.exists(dirs[1])):
		print "Train/Test files not detected, creating now..."
		config = ConfigParser.ConfigParser()
		config.read(FLAGS.config_file)
		max_num_lines = int(config.get("max_data_sizes", "num_lines"))
		max_target_size = int(config.get("max_data_sizes", "max_target_length"))
		max_source_size = int(config.get("max_data_sizes", "max_source_length"))
		data_processor = data_utils.DataProcessor(FLAGS.vocab_size,
												  FLAGS.raw_data_dir,
											      FLAGS.data_dir,
												  FLAGS.train_frac,
												  FLAGS.tokenizer,
												  max_num_lines,
												  max_target_size,
												  max_source_size)
		data_processor.run()

	assert FLAGS.plot_histograms or FLAGS.plot_scatter, "Must choose at least one plot!"
	source_lengths = []
	target_lengths = []
	count = 0
	for i in range(len(dirs)):
		if "test" in dirs[i]:
			source_path = os.path.join(dirs[i], "data_source_test.txt")
			target_path = os.path.join(dirs[i], "data_target_test.txt")
		else:
			source_path = os.path.join(dirs[i], "data_source_train.txt")
			target_path = os.path.join(dirs[i], "data_target_train.txt")
		with tf.gfile.GFile(source_path, mode="r") as source_file:
			with tf.gfile.GFile(target_path, mode="r") as target_file:
				source, target = source_file.readline(), target_file.readline()
				counter = 0
				while source and target:
					counter += 1
					if counter % 100000 == 0:
						print("  reading data line %d" % counter)
						sys.stdout.flush()
					num_source_ids = len(source.split())
					source_lengths.append(num_source_ids)
					#plus 1 for EOS token
					num_target_ids = len(target.split()) + 1
					target_lengths.append(num_target_ids)
					source, target = source_file.readline(), target_file.readline()
	if FLAGS.plot_histograms:
		plot_histo_lengths("target lengths", target_lengths)
		plot_histo_lengths("source_lengths", source_lengths)
	if FLAGS.plot_scatter:
		plot_scatter_lengths("target vs source length", "source length",
			"target length", source_lengths, target_lengths)


def plot_scatter_lengths(title, x_title, y_title, x_lengths, y_lengths):
	plt.scatter(x_lengths, y_lengths)
	plt.title(title)
	plt.xlabel(x_title)
	plt.ylabel(y_title)
	plt.ylim(0, max(y_lengths))
	plt.xlim(0,max(x_lengths))
	plt.show()

def plot_histo_lengths(title, lengths):
	mu = np.std(lengths)
	sigma = np.mean(lengths)
	x = np.array(lengths)
	n, bins, patches = plt.hist(x,  num_bins, facecolor='green', alpha=0.5)
	y = mlab.normpdf(bins, mu, sigma)
	plt.plot(bins, y, 'r--')
	plt.title(title)
	plt.xlabel("Length")
	plt.ylabel("Number of Sequences")
	plt.xlim(0,max(lengths))
	plt.show()


if __name__=="__main__":
	main()
