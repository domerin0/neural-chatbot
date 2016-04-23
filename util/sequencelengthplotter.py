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
import sys
import numpy as np
import os
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "data/", "data directory")
flags.DEFINE_boolean("plot_histograms", True, "Plot histograms of sequence lengths")
flags.DEFINE_boolean("plot_scatter", True, "Make scatter plot of target vs. source length")

num_bins = 50

def main():
	assert FLAGS.plot_histograms or FLAGS.plot_scatter, "Must choose at least one plot!"
	source_lengths = []
	target_lengths = []
	count = 0
	dirs = [os.path.join(FLAGS.data_dir, "train/"),
	os.path.join(FLAGS.data_dir, "test/")]
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
		plotHistoLengths("target lengths", target_lengths)
		plotHistoLengths("source_lengths", source_lengths)
	if FLAGS.plot_scatter:
		plotScatterLengths("target vs source length", "source length",
			"target length", source_lengths, target_lengths)


def plotScatterLengths(title, x_title, y_title, x_lengths, y_lengths):
	plt.scatter(x_lengths, y_lengths)
	plt.title(title)
	plt.xlabel(x_title)
	plt.ylabel(y_title)
	plt.ylim(0, max(y_lengths))
	plt.xlim(0,max(x_lengths))
	plt.show()

def plotHistoLengths(title, lengths):
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
