import os
import pickle


def restoreHyperParams(file_path):
    '''
    Restore serialized hyper params
    Inputs:

    file_path: path to checkpoint dir
    '''
    file_path = os.path.join(file_path, "hyperparams.p")
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def saveHyperParameters(checkpoint_dir, FLAGS):
    '''
    Serialize hyper parameters fopr checkpoint restoration later
    '''
    dic = {"vocab_size" : FLAGS.vocab_size,
        "hidden_size" : FLAGS.hidden_size,
        "dropout" : FLAGS.dropout,
        "grad_clip" : FLAGS.grad_clip,
        "num_layers" : FLAGS.num_layers,
        "learning_rate" : FLAGS.learning_rate,
        "lr_decay_factor" : FLAGS.lr_decay_factor}
    path = os.path.join(checkpoint_dir, "hyperparams.p")
    with open(path, 'wb') as handle:
        pickle.dump(dic, handle)
