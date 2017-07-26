# neural-chatbot
A chatbot based on seq2seq architecture.

### About

This is the successor to my previous torch project 'seq2seq-chatbot'. Portions of this project were directly adapted from the tensorflow translation example.

This is based off the research in these papers: [Sutskever et al., 2014.](http://arxiv.org/abs/1409.3215) and [Vinyals & Le, 2015.](http://arxiv.org/pdf/1506.05869v1.pdf)

### Dependencies

1. Python 3
2. [TensorFlow 1.2](https://www.tensorflow.org/install/)
3. (optional) matplotlib `$ pip install matplotlib`

### How to use

To use your own data read Data Format, otherwise to use included [Cornell Movie Dialogues Corpur](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) data:

`$ python train.py`

#### Data Format

If you wish to provide your own data use the format below to provide your own data.

#### Data Format

Each continuous conversation must be in it's own text file. Each line of the text files is a response to the previous line. The exception of course is that the first line of the text file is the conversation start. After entering your own data, you can run it with:

`$ python sequencelengthplotter.py --data_dir="path/to/text_files"`

The options for the above command are:

|  Name | Type  | Description  |
|:---:|:---:|:---:|
| data_dir  | string  | path to raw data files  |
| plot_histograms  |  boolean | plots histograms for  target and source sequences |
| plot_scatter  | boolean  |  plot x-y scatter plot for target length vs. source length |

in `buckets.cfg` you can then modify the bucket values accordingly. You can add or remove buckets. All bucket values in the [buckets] subheading will be parsed in. Each line under [buckets] should be of the format:

``bucket_name: source_length,target_length``

In the same configuration file the data settings can also be changed under the [max_data_sizes] subheading.

|  Name | Type  | Description  |
|:--------:|:--------:|:--------:|
| num_lines  | int  | number of lines in conversation to go back  |
| max_target_length  |  int | max length of target sequences |
| max_source_length  | int  |  max length of source sequences |

A configuration file was used because it was a mess trying to find out how to pass bucket values via command line. This seemed like a half-way decent solution. It also enables (in the future) `sequencelengthplotter.py` the ability to autonomously change these values without requiring any user input.

#### Training Network

Once you are satisfied with bucket sizes you can then run the optimizer. This will also clean and process your raw text data. To do this:

`$ python train.py --raw_data_dir="data/cornell_lines"`

There are several options that can be employed:

|   Name               | Type          |     Description                            |
| :-------------------:|:-------------:|:-------------------------------------------|
| hidden_size          | int           | number of hidden units in hidden layers    |
| num_layers           | int           |   number of hidden layers                  |
| batch_size           | int           |    size of batchs in training              |
| max_epoch            | int           |    max number of epochs to train for       |
| learning_rate        | float         |    beggining learning rate                 |
| steps_per_checkpoint | int           |    number of steps before running test set |
| lr_decay_factor      | float         |    factor by which to decay learning rate  |
| batch_size           | int           |    size of batchs in training              |
| checkpoint_dir       | string        |    directory to store/restore checkpoints  |
| dropout              | float         | probability of hidden inputs being removed |
| grad_clip            | int           |    max gradient norm                       |
| max_train_data_size  | int           |    Use a subset of available data to train |
| vocab_size           | int           |    max vocab size                          |
| train_frac           | int           |    percentage of data to use for training (rest is used for testing)   |
| raw_data_dir         | int           |    raw conversation text files stored here |
| data_dir             | int           |    Directory data processor will store train/test/vocab files          |
| max_source_length    | int           |  How long the source sentences can be at most |
| max_target_length    | int           |  How long the target sentences can be at most |
| convo_limits         | int           | How far back the bot's memory should go for the conversation |

#### Tensorboard Usage

Tensorboard not yet implemented.

#### Sampling output

** This still needs to be tested since the update to TF 1.2

To have a real conversation with your bot you can begin an interactive prompt by doing:

`$ python sample.py --checkpoint_dir"path/to/checkpointdirectory" --data_dir="path/to/datadirectory"`

Summary of options below:

|  Name | Type  | Description  |
|:---:|:---:|:---:|
| checkpoint_dir  | string  | path to saved checkpoint  |
| data_dir  |  string | path to directory where vocab file is |

### Results

Coming soon.

### Future Features

- Downloadable pretrained model
