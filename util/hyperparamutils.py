import os
import pickle


def restore_hyper_params(file_path):
    '''
    Restore serialized hyper params
    Inputs:

    file_path: path to checkpoint dir
    '''
    file_path = os.path.join(file_path, "hyperparams.p")
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def save_hyper_params(checkpoint_dir, FLAGS, buckets, conversation_limits):
    '''
    Serialize hyper parameters fopr checkpoint restoration later
    Inputs:
    checkpoint_dir
    FLAGS:
    conversation_limits: list of 3 ints [max_source_size, max_target_size, conversation_history]
    '''
    dic = {"vocab_size" : FLAGS.vocab_size,
           "enc_hidden_size" : FLAGS.hidden_size,
           "dec_hidden_size" : FLAGS.dec_hidden_size,
           "dropout" : FLAGS.dropout,
           "grad_clip" : FLAGS.grad_clip,
           "num_layers" : FLAGS.num_layers,
           "learning_rate" : FLAGS.learning_rate,
           "lr_decay_factor" : FLAGS.lr_decay_factor,
           "max_source_length": conversation_limits[0],
           "max_target_length":conversation_limits[1],
           "conversation_history": conversation_limits[2]
           }

    path = os.path.join(checkpoint_dir, "hyperparams.p")
    with open(path, 'wb') as handle:
        pickle.dump(dic, handle)
