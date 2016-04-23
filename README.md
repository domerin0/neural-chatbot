# neural-chatbot
A chatbot based on seq2seq architecture.

### About

This is the successor to my previous torch project 'seq2seq-chatbot'. Leveraging the tensorflow translation example and also


### Dependencies

1. Python 2.7
2. [TensorFlow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
3. (optional) matplotlib `$ pip install matplotlib`

### How to use

#### Getting data

To use this (currently) you will have to provide your own data. This is temporary until I add download functionality to an 
existing online dataset. Your data should be in a format where each succeeding line in a text file is in response to the one
above it. You also should have one text file per conversation. In order to help you choose bucket sizes I've written a script
to plot the sequence lengths of the source and target sequences. To use this do:

`$ python util/sequencelengthplotter.py --data_dir="path/to/raw_data_files"`

The options for the above command are:

|  Name | Type  | Description  |
|---|---|---|
| data_dir  | string  | path to raw data files  |
| plot_histograms  |  boolean | plots histograms for  target and source sequences |
| plot_scatter  | boolean  |  plot x-y scatter plot for target length vs. source length |

in `train.py` you can then modify the bucket values accordingly. I do plan on making this an easier process,

Once you are satisfied with bucket sizes you can then run the optimizer. To do this:

`$ python train.py`

There are several options that can be employed:



### Results

So far using a Titan X, after about 12 hours of training, I achieve a perplexity of ~30 on a 'relatively small' network. 
Results with human test so far haven't been too great, I am trying to find the right set of parameters to get something 'ok', 
that can be trained in 24 hours or less on my GPU. Reults will be added here when found.
