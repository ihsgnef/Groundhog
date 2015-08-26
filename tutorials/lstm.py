"""
Simple implementation of LSTM language model
"""
from groundhog.datasets import LMIterator
from groundhog.trainer.SGD_momentum import SGD as SGD_m
from groundhog.trainer.SGD import SGD
from groundhog.mainLoop import MainLoop
from groundhog.layers import MultiLayer, \
       RecurrentMultiLayer, \
       RecurrentMultiLayerInp, \
       RecurrentMultiLayerShortPath, \
       RecurrentMultiLayerShortPathInp, \
       RecurrentMultiLayerShortPathInpAll, \
       SoftmaxLayer, \
       LastState,\
       UnaryOp, \
       Operator, \
       Shift, \
       GaussianNoise, \
       SigmoidLayer
from groundhog.layers import maxpool, \
        maxpool_ntimes, \
        last, \
        last_ntimes,\
        tanh, \
        sigmoid, \
        rectifier,\
        hard_sigmoid, \
        hard_tanh
from groundhog.models import LM_Model
from theano.sandbox.scan import scan

import numpy as np
import theano
import theano.tensor as TT

linear = lambda x:x

theano.config.allow_gc = False

def get_text_data(state):
    def out_format (x, y, r):
        return {'x':x, 'y' :y, 'reset': r}
    def out_format_valid (x, y, r):
        return {'x':x, 'y' :y, 'reset': r}

    train_data = LMIterator(
            batch_size=state['bs'],
            path = state['path'],
            stop=-1,
            seq_len = state['seqlen'],
            mode="train",
            chunks=state['chunks'],
            shift = state['shift'],
            output_format = out_format,
            can_fit=True)

    valid_data = LMIterator(
            batch_size=state['bs'],
            path=state['path'],
            stop=-1,
            use_infinite_loop=False,
            allow_short_sequences = True,
            seq_len= state['seqlen'],
            mode="valid",
            reset =state['reset'],
            chunks=state['chunks'],
            shift = state['shift'],
            output_format = out_format_valid,
            can_fit=True)

    test_data = LMIterator(
            batch_size=state['bs'],
            path = state['path'],
            stop=-1,
            use_infinite_loop=False,
            allow_short_sequences=True,
            seq_len= state['seqlen'],
            mode="test",
            chunks=state['chunks'],
            shift = state['shift'],
            output_format = out_format_valid,
            can_fit=True)
    if 'wiki' in state['path']:
        test_data = None
    return train_data, valid_data, test_data

def create_embedding_layers(state)
    # create embedding layers for 4 gates
    # rank_n_approx determines whether to approximate the
    # embedding with in a lower dimension

    # activation should be x : x
    # because based on GroundHog's design,
    # the actual activation is handled by LSTM layer all together

    input_embs  = [lambda x : 0] * state['stack_number']
    update_embs = [lambda x : 0] * state['stack_number']
    forget_embs = [lambda x : 0] * state['stack_number']
    output_embs = [lambda x : 0] * state['stack_number']

    default_kwargs = {
            n_in           : state['n_in'],
            n_hids         : eval(state['emb_nhids']),

            activation     : eval(state['emb_activ']),
            init_fn        : 'sample_weights_classic',

            rank_n_approx  : state['rank_n_approx'],

            scale          : state['emb_sparse'],
            bias_scale     : eval(state['emb_bias']) 

            weight_noise   : state['emb_weight_noise'] 
            learn_bias     : True,
            }

    for level in xrange(state['stack_number']):
        input_embs_kwargs = default_kwargs.update(name = 'input_emb')
        input_embs[level] = MultiLayer(
                rng, input_embs_kwargs)

        update_embs_kwargs = default_kwargs.update(name = 'update_emb')
        update_embs[level] = MultiLayer(
                rng, update_embs_kwargs)

        forget_embs_kwargs = default_kwargs.update(name = 'forget_emb')
        forget_embs[level] = MultiLayer(
                rng, forget_embs_kwargs)

        output_embs_kwargs = default_kwargs.update(name = 'output_emb')
        output_embs[level] = MultiLayer(
                rng, output_embs_kwargs)

    return input_embs, update_embs, forget_embs, output_embs


def create_intermediate_layers(state)
    # create embedding layers for 4 gates:
    # intermediate layers take previous hidden layer as input

    # activation should be x : x
    # because based on GroundHog's design,
    # the actual activation is handled by LSTM layer all together

    input_ints  = [0] * state['stack_number']
    update_ints = [0] * state['stack_number']
    forget_ints = [0] * state['stack_number']
    output_ints = [0] * state['stack_number']

    default_kwargs = {
            n_in           : state['hid_'],
            n_hids         : eval(state['int_nhids']),

            activation     : eval(state['int_activ']),
            init_fn        : 'sample_weights_classic',

            rank_n_approx  : False

            weight_noise   : state['weight_noise'],

            scale          : state['int_sparse'],
            bias_scale     : eval(state['int_bias']),

            learn_bias     : True
            }

    for level in xrange(state['stack_number']):
        input_ints_kwargs = default_kwargs.update(name = 'input_int')
        input_ints[level] = MultiLayer(
                rng, input_ints_kwargs)

        update_ints_kwargs = default_kwargs.update(name = 'update_int')
        update_ints[level] = MultiLayer(
                rng, update_ints_kwargs)

        forget_ints_kwargs = default_kwargs.update(name = 'forget_int')
        forget_ints[level] = MultiLayer(
                rng, forget_ints_kwargs)

        output_ints_kwargs = default_kwargs.update(name = 'output_int')
        output_ints[level] = MultiLayer(
                rng, output_ints_kwargs)

    return input_embs, update_embs, forget_embs, output_embs

def jobman(state, channel):
    rng = np.random.RandomState(state['seed'])

    if state['chunks'] == 'words':
        state['n_in'] = 10000
        state['n_out'] = 10000
    else
        state['n_in'] = 50
        state['n_out'] = 50

    train_data, valid_data, test_data = get_text_data(state)

    x = TT.lvector('x')
    y = TT.lvector('y')
    h0 = theano.shared(np.zeros((eval(state['nhids'])[-1],), dtype='float32'))

    input_embs, update_embs, forget_embs, output_embs = create_embedding_layers(state)

    input_ints, update_ints, forget_ints, output_ints = create_intermediate_layers(state)



def main():
    state = {}
    datadir = './'

    state['seqlen'] = 100
    state['path']= datadir + "tmp_data.npz"
    state['dictionary']= datadir + "tmp_data_dict.npz"
    state['chunks'] = 'words'
    state['seed'] = 123
    state['on_nan'] = 'warn'
    state['reset'] = -1
    state['minlr'] = float(5e-7)

    # hidden layer stack size
    state['stack_number'] = 1

# layers

