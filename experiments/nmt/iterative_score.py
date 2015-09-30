#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys

import numpy

import experiments.nmt
from experiments.nmt import\
    RNNEncoderDecoder,\
    prototype_state,\
    parse_input

from experiments.nmt.numpy_compat import argpartition


class ScoreMaker(object):
    
    def __init__(self, enc_dec):
        self.enc_dec = enc_dec
        state = self.enc_dec.state
        self.eos_id = state['null_sym_target']
        self.unk_id = state['null_sym_target']

    def compile(self):
        self.comp_repr = self.enc_dec.create_representation_computer()
        self.comp_init_states = self.enc_dec.create_initializers()
        self.comp_next_probs = self.enc_dec.create_next_probs_computer()
        self.comp_next_states = self.enc_dec.create_next_states_computer()

    def score(lm_model, seq_src, seq_trg, idict_src, idict_trg,
              comp_repr, comp_init_state, comp_next_prob, comp_next_state):
        pass

    

def indices_to_words(i2w, seq):
    sen = []
    for k in xrange(len(seq)):
        if i2w[seq[k]] == '<eol>':
            break
        sen.append(i2w[seq[k]])
    return sen


def parse_args():
    parser = argparse.ArgumentParser(
            "Sample translations from a translation model")
    parser.add_argument("--state", required=True, help="State to use")
    parser.add_argument("--source", help="File of source sentences")
    parser.add_argument("--target", help="File of target sentences")
    parser.add_argument("--trans", help="File to save translations in")
    parser.add_argument("model_path", help="Path to the model")
    return parser.parse_args()


def main():
    args = parse_args()
    state = prototype_state()

    with open(args.state) as src:
        state.update(cPickle.load(src))

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)

    scoreMaker = ScoreMaker(enc_dec)
    ScoreMaker.compile()

    indx_word_src = cPickle.load(open(state['word_indx'],'rb'))
    indx_word_trg = cPickle.load(open(state['word_indx_trgt'],'rb'))

    idict_src = cPickle.load(open(state['indx_word'],'r'))
    idict_trg = cPickle.load(open(state['indx_word_target'],'r'))

    fsrc = open(args.source, 'r')
    ftrg = open(args.target, 'r')
    for srcline, trgline in zip(fsrc, ftrg):
        src_seqin = srcline.strip()
        trg_seqin = trgline.strip()
        src_seq, src_parsed_in = parse_input(state, 
                                             indx_word_src, 
                                             src_seqin, 
                                             idx2word=idict_src)
        trg_seq, trg_parsed_in = parse_input(state, 
                                             indx_word_trg, 
                                             trg_seqin, 
                                             idx2word=idict_trg)
        print "Parsed Input:", src_parsed_in

        ScoreMaker.score(lm_model, src_seq, trg_seq, idict_src, idict_trg)

    fsrc.close()
    ftrg.close()


if __name__ == "__main__":
    main()
