"""
a LSTM language model implemented in Encoder-Decoder framework
the encoder is a standard LSTM
the decoder is a copy of encoder's output
"""

import numpy as np
import logging
import pprint
import operator
import itertools

import theano
import theano.tensor as TT
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.layers import\
        Layer,\
        MultiLayer,\
        SoftmaxLayer,\
        HierarchicalSoftmaxLayer,\
        LSTMLayer, \
        RecurrentLayer,\
        RecursiveConvolutionalLayer,\
        UnaryOp,\
        Shift,\
        LastState,\
        DropOp,\
        Concatenate
from groundhog.models import LM_Model
from groundhog.datasets import PytablesBitextIterator
from groundhog.utils import sample_zeros, sample_weights_orth, init_bias, sample_weights_classic
import groundhog.utils as utils

logger = logging.getLogger(__name__)

def create_padded_batch(state, x, y, return_dict=False):
    """A callback given to the iterator to transform data in suitable format

    :type x: list
    :param x: list of np.array's, each array is a batch of phrases
        in some of source languages

    :type y: list
    :param y: same as x but for target languages

    :param new_format: a wrapper to be applied on top of returned value

    :returns: a tuple (X, Xmask, Y, Ymask) where
        - X is a matrix, each column contains a source sequence
        - Xmask is 0-1 matrix, each column marks the sequence positions in X
        - Y and Ymask are matrices of the same format for target sequences
        OR new_format applied to the tuple

    Notes:
    * actually works only with x[0] and y[0]
    * len(x[0]) thus is just the minibatch size
    * len(x[0][idx]) is the size of sequence idx
    """
    
    mx = state['seqlen']
    my = state['seqlen']
    if state['trim_batches']:
        # Similar length for all source sequences
        mx = np.minimum(state['seqlen'], max([len(xx) for xx in x[0]]))+1
        # Similar length for all target sequences
        my = np.minimum(state['seqlen'], max([len(xx) for xx in y[0]]))+1

    # Batch size
    n = x[0].shape[0]

    X = np.zeros((mx, n), dtype='int64')
    Y = np.zeros((my, n), dtype='int64')
    Xmask = np.zeros((mx, n), dtype='float32')
    Ymask = np.zeros((my, n), dtype='float32')

    # Fill X and Xmask
    for idx in xrange(len(x[0])):
        # Insert sequence idx in a column of matrix X
        if mx < len(x[0][idx]):
            X[:mx, idx] = x[0][idx][:mx]
        else:
            X[:len(x[0][idx]), idx] = x[0][idx][:mx]

        # Mark the end of phrase
        if len(x[0][idx]) < mx:
            X[len(x[0][idx]):, idx] = state['null_sym_source']

        # Initialize Xmask column with ones in all positions that
        # were just set in X
        Xmask[:len(x[0][idx]), idx] = 1.
        if len(x[0][idx]) < mx:
            Xmask[len(x[0][idx]), idx] = 1.

    # Fill Y and Ymask in the same way as X and Xmask in the previous loop
    for idx in xrange(len(y[0])):
        Y[:len(y[0][idx]), idx] = y[0][idx][:my]
        if len(y[0][idx]) < my:
            Y[len(y[0][idx]):, idx] = state['null_sym_target']
        Ymask[:len(y[0][idx]), idx] = 1.
        if len(y[0][idx]) < my:
            Ymask[len(y[0][idx]), idx] = 1.

    null_inputs = np.zeros(X.shape[1])

    # We say that an input pair is valid if both:
    # - either source sequence or target sequence is non-empty
    # - source sequence and target sequence have null_sym ending
    # Why did not we filter them earlier?
    for idx in xrange(X.shape[1]):
        if np.sum(Xmask[:,idx]) == 0 and np.sum(Ymask[:,idx]) == 0:
            null_inputs[idx] = 1
        if Xmask[-1,idx] and X[-1,idx] != state['null_sym_source']:
            null_inputs[idx] = 1
        if Ymask[-1,idx] and Y[-1,idx] != state['null_sym_target']:
            null_inputs[idx] = 1

    valid_inputs = 1. - null_inputs

    # Leave only valid inputs
    X = X[:,valid_inputs.nonzero()[0]]
    Y = Y[:,valid_inputs.nonzero()[0]]
    Xmask = Xmask[:,valid_inputs.nonzero()[0]]
    Ymask = Ymask[:,valid_inputs.nonzero()[0]]
    if len(valid_inputs.nonzero()[0]) <= 0:
        return None

    # Unknown words
    X[X >= state['n_sym_source']] = state['unk_sym_source']
    Y[Y >= state['n_sym_target']] = state['unk_sym_target']

    if return_dict:
        return {'x' : X, 'x_mask' : Xmask, 'y': Y, 'y_mask' : Ymask}
    else:
        return X, Xmask, Y, Ymask

def get_batch_iterator(state):

    class Iterator(PytablesBitextIterator):

        def __init__(self, *args, **kwargs):
            PytablesBitextIterator.__init__(self, *args, **kwargs)
            self.batch_iter = None
            self.peeked_batch = None

        def get_homogenous_batch_iter(self):
            while True:
                k_batches = state['sort_k_batches']
                batch_size = state['bs']
                data = [PytablesBitextIterator.next(self) for k in range(k_batches)]
                x = np.asarray(list(itertools.chain(*map(operator.itemgetter(0), data))))
                y = np.asarray(list(itertools.chain(*map(operator.itemgetter(1), data))))
                lens = np.asarray([map(len, x), map(len, y)])
                order = np.argsort(lens.max(axis=0)) if state['sort_k_batches'] > 1 \
                        else np.arange(len(x))
                for k in range(k_batches):
                    indices = order[k * batch_size:(k + 1) * batch_size]
                    batch = create_padded_batch(state, [x[indices]], [y[indices]],
                            return_dict=True)
                    if batch:
                        yield batch

        def next(self, peek=False):
            if not self.batch_iter:
                self.batch_iter = self.get_homogenous_batch_iter()

            if self.peeked_batch:
                # Only allow to peek one batch
                assert not peek
                logger.debug("Use peeked batch")
                batch = self.peeked_batch
                self.peeked_batch = None
                return batch

            if not self.batch_iter:
                raise StopIteration
            batch = next(self.batch_iter)
            if peek:
                self.peeked_batch = batch
            return batch

    train_data = Iterator(
        batch_size=int(state['bs']),
        target_file=state['target'][0],
        source_file=state['source'][0],
        can_fit=False,
        queue_size=1000,
        shuffle=state['shuffle'],
        use_infinite_loop=state['use_infinite_loop'],
        max_len=state['seqlen'])
    return train_data


def prefix_lookup(state, p, s):
    if '%s_%s'%(p,s) in state:
        return state['%s_%s'%(p, s)]
    return state[s]


class LongShortTermMemoryBase(Object):

    def _create_embedding_layers(self):
        logger.debug("_create_embedding_layers")

        # use n rank approximation for input embeddings
        # through a 3-layer subnetwork

        self.approx_embedder = MultiLayer(
            self.rng,
            n_in=self.state['n_sym_source']
                if self.prefix.find("enc") >= 0
                else self.state['n_sym_target'],
            n_hids=[self.state['rank_n_approx']],
            activation=[self.state['rank_n_activ']],
            name='{}_approx_embdr'.format(self.prefix),
            **self.default_kwargs)

        # we have 4 embeddings for each word in each level
        # input gate
        # update (new memory) gate 
        # forget gate
        # exposure gate

        self.input_embedders = [lambda x : 0] * self.num_levels
        self.update_embedders = [lambda x : 0] * self.num_levels
        self.forget_embedders = [lambda x : 0] * self.num_levels
        self.output_embedders = [lambda x : 0] * self.num_levels

        embedder_kwargs = dict(self.default_kwargs)
        embedder_kwargs.update(dict(
            n_in = self.state['rank_n_approx'],
            n_hids = [self.state['dim'] * self.state['dim_mult']],
            activation = ['lambda x : x']))

        # expand the network num_levels times through time
        
        for level in xrange(self.num_levels):
            self.input_embedders[level] = MultiLayer(
                    self.rng,
                    name = '{}_input_embdr_{}'.format(self.prefix, level),
                    **embedder_kwargs)

            self.update_embedders[level] = MultiLayer(
                    self.rng,
                    name = '{}_update_embdr_{}'.format(self.prefix, level),
                    # learn_bias = False,
                    **embedder_kwargs)

            self.forget_embedders[level] = MultiLayer(
                    self.rng,
                    name = '{}_update_embdr_{}'.format(self.prefix, level),
                    # learn_bias = False,
                    **embedder_kwargs)

            self.output_embedders[level] = MultiLayer(
                    self.rng,
                    name = '{}_update_embdr_{}'.format(self.prefix, level),
                    # learn_bias = False,
                    **embedder_kwargs)

    def _create_inter_level_layers(self):
        logger.debug("_create_inter_level_layers")
        inter_level_kwargs = dict(self.default_kwargs)
        inter_level_kwargs.update(
                n_in = self.state['dim'],
                n_hids = self.state['dim'] * self.state['dim_mult'],
                activation = ['lambda x : x'])

        self.input_inters = [0] * self.num_levels
        self.update_inters = [0] * self.num_levels
        self.forget_inters = [0] * self.num_levels
        self.output_inters = [0] * self.num_levels

        # these intermediate gate takes previous hidden state
        # so start from time 1 not 0
        for level in xrange(1, self.num_levels):
            self.input_inters[level] = MultiLayer(
                    sefl.rng,
                    name = '{}_input_inter_{}'.format(self.prefix, level),
                    **inter_level_kwargs)

            self.update_inters[level] = MultiLayer(
                    sefl.rng,
                    name = '{}_input_inter_{}'.format(self.prefix, level),
                    **inter_level_kwargs)

            self.forget_inters[level] = MultiLayer(
                    sefl.rng,
                    name = '{}_input_inter_{}'.format(self.prefix, level),
                    **inter_level_kwargs)

            self.output_inters[level] = MultiLayer(
                    sefl.rng,
                    name = '{}_input_inter_{}'.format(self.prefix, level),
                    **inter_level_kwargs)

    def _create_transition_layers(self):
        logger.debug('_create_transition_layers')
        self.transitions = []
        for level in xrange(self.num_levels):
            self.transitions.append(LSTMLayer(
                self.rng,
                n_hids = self.stats['dim'],
                activation = prefix_lookup(self.state, self.prefix, 'activ'),
                bias_scale = self.state['bias'],
                inint_fn = (self.state['rec_weight_init_fn'],
                    if not self.skip_init
                    else 'sample_zeros'),
                scale = prefix(self.state, self.prefix, 'rec_weight_scale'),
                weight_noise = self.state['weight_noise_rec'],
                dropout = self.state['dropout_rec'],
                # gating is False for LSTM
                gating = prefix_lookup(self.state, self.prefix, 'rec_gating'),
                # reseting is False for LSTM
                reseting = prefix_lookup(self.state, self.prefix, 'rec_resetin'),
                name = '{}_transition_{}'.format(self.prefix, level)))

class Encoder(LongShortTermMemoryBase):

    def __init__(self, state, rng, prefix = 'enc', skip_init = False):
        self.state = state
        self.rng = rng
        self.prefix = prefix
        self.skip_init = skip_init
        
        self.num_levels = self.state['encoder_stack']

    
    def create_layers(self):
        self.default_kwargs = dict(
                init_fn = self.state['weight_init_fn']
                    if not self.skip_init
                    else 'sample_zeros',
                weight_noise = self.state['weight_noise'],
                scale = self.state['weight_scale'])

        self._create_embedding_layers()
        self._create_transition_layers()
        self._create_inter_level_layers()
        self._create_representation_layers()

    def _create_representation_layers(self):
        logger.debug('_create_representation_layers')
