import numpy as np
import copy
import operator
import pickle
import kneser_ney

from tensorpack import *

import pdb

try:
    from .cfgs.config import cfg
except Exception:
    from cfgs.config import cfg


def confirm_prefix(container, prefix):
    if prefix not in container.keys():
        container[prefix] = [0, 0]

def load_lm():
    orders = [1, 2, 3]
    files = [open('lm_%d.model' % e, 'rb') for e in orders]
    lms = [pickle.load(e) for e in files]
    return lms

def lm_prob(prefix, lms):
    # get rid of the start symbol '@'
    prefix = prefix[1:]
    words = prefix.split(' ')
    real_words = []
    for word in words:
        if word != "":
            real_words.append(word)
    words = real_words
    if len(words) == 0:
        return 1
    lm_type = 0
    if lm_type == 1:
        if len(words) == 1:
            prob = lms[0].logprob(tuple(words))
        elif len(words) == 2:
            prob = lms[1].logprob(tuple(words))
        else:
            prob = lms[2].logprob(tuple(words[-3:]))
        if prob == None:
            return 0
        else:
            return 10 ** prob
    elif lm_type == 2:
        if len(words) == 1:
            ngram = ((lm[2].start_pad_symbol,) * 2 + tuple(words))
        elif len(words) == 2:
            ngram = ((lms[2].start_pad_symbol,) * 1 + tuple(words))
        else:
            ngram = tuple(words[-3:])
        prob = lms[2].logprob(ngram)
        if prob == None:
            return 0
        else:
            return 10 ** prob
    else:
        return 1

def decode_with_lm(logits, lms, beam_width):
    '''
    Decode the network output to a list of words
    Argments:
        logits: the network output as an array, with shape of (sequence length, alphabet size plus 1)
        lm: the language model
        k: beam width
    Return:
        the top k output and their prob.
    '''

    beta = 0

    # the initial prefix set only has an empty string
    # two prob. for each prefix. they are the prob. for each prefix ending in blank (first one) or not ending in blank (second one) 
    A_prev = {"@": [1, 0]}         # the symbol @ is not in alphabet and we use it as the start symbol
    A_prev_all = {"@": [1, 0]}
    for t in range(logits.shape[0]):
        # for each time step
        A_next = {}
        prob_next = []
        for prefix in A_prev.keys():
            # for every prefix currently in A_prev
            for char_idx in range(len(cfg.dictionary) + 1):
                # for each character in the alphabet
                if char_idx == len(cfg.dictionary):
                    # the character is the blank label
                    confirm_prefix(A_next, prefix)
                    A_next[prefix][0] += logits[t, char_idx] * (A_prev[prefix][0] + A_prev[prefix][1])
                else:
                    # the charactor is not the blank label
                    prefix_plus = prefix + cfg.dictionary[char_idx]
                    confirm_prefix(A_next, prefix_plus)
                    confirm_prefix(A_next, prefix)
                    if cfg.dictionary[char_idx] == prefix[-1]:
                        # if the new char is the same as the last char in the prefix
                        A_next[prefix_plus][1] += logits[t, char_idx] * A_prev[prefix][0]
                        A_next[prefix][1] += logits[t, char_idx] * A_prev[prefix][1]
                    elif char_idx == cfg.dictionary.index(' '):
                        # if the new char is a space
                        A_next[prefix_plus][1] = lm_prob(prefix_plus, lms) * logits[t, char_idx] * (A_prev[prefix][0] + A_prev[prefix][1])
                    else:
                        A_next[prefix_plus][1] = logits[t, char_idx] * (A_prev[prefix][0] + A_prev[prefix][1])
                    if prefix_plus not in A_prev.keys() and prefix_plus in A_prev_all.keys():
                        A_next[prefix_plus][0] += logits[t, len(cfg.dictionary)] * (A_prev_all[prefix_plus][0] + A_prev_all[prefix_plus][1])
                        A_next[prefix_plus][1] += logits[t, char_idx] * A_prev_all[prefix_plus][1]
        # word insertion item and prob. normalization should be done here
        prefix_str_ary = [e[0] for e in A_next.items()]
        prefix_str_ary = [e[1:] for e in prefix_str_ary]
        words_num = []
        for prefix_str in prefix_str_ary:
            words = prefix_str.split(' ')
            word_num = 0
            for word in words:
                if word != "":
                    word_num += 1
            words_num.append(word_num)

        A_next_p = { }
        for k, v in A_next.items():
            words = k.split(' ')
            word_num = 0
            for word in words:
                if word != "":
                    word_num += 1
            A_next_p[k] = [v[0] * word_num ** beta, v[1] * word_num ** beta]
        A_next = A_next_p

        # select most k probable prefixes
        temp = [[e[0], sum(e[1])] for e in A_next.items()]
        temp = sorted(temp, key=operator.itemgetter(1), reverse=True)
        temp = temp[:beam_width]
        temp = [e[0] for e in temp]
        A_prev = { }
        for k in temp:
            A_prev[k] = copy.deepcopy(A_next[k])
        A_prev_all = copy.deepcopy(A_next)
        best_prev = temp[0][1:]
    return best_prev

if __name__ == "main":

    # logits is sequence_len x label_size
    logits = np.random.rand(100, len(cfg.dictionary) + 1)
    for t in range(100):
        e = np.exp(logits[t])
        logits[t] = e / np.sum(e)
    
    beam_width = 100

    result = decode_with_lm(logits, None, beam_width)

    result

