"""Utility functions."""
import progressbar

import numpy

from chainer import cuda
from chainer.dataset import convert

from gensim.models import KeyedVectors


# speical symbols
PAD = -1
UNK = 0
EOS = 1


def fact_pad_concat_convert(fact_batch, device, test=False):
    """

    Args:
        fact_batch: List of tuples of heads, relations, tails, and labels.
        device: Device ID to which an array is sent.
        test: If it is test, this flag should be true.

    Returns:
        Tuple of Converted array.

    """
    hs, rs, ts, ys = zip(*fact_batch)

    h_block = convert.concat_examples(hs, device, padding=PAD)
    r_block = convert.concat_examples(rs, device, padding=PAD)
    t_block = convert.concat_examples(ts, device, padding=PAD)
    if test is True:
        y_block = convert.concat_examples(ys, device, padding=PAD)
    else:
        # add negative example
        xp = cuda.get_array_module(h_block)
        h_block_neg = h_block.copy()
        r_block_neg = r_block.copy()
        t_block_neg = t_block.copy()
        xp.random.shuffle(h_block_neg)
        xp.random.shuffle(r_block_neg)
        xp.random.shuffle(t_block_neg)
        h_block = xp.concatenate((h_block, h_block_neg))
        r_block = xp.concatenate((r_block, r_block_neg))
        t_block = xp.concatenate((t_block, t_block_neg))
        y_block = convert.concat_examples(
            xp.concatenate(
                (xp.ones(len(hs), 'i'), xp.zeros(len(hs), 'i')),
            ),
            device,
            padding=PAD
        )
    return (h_block, r_block, t_block, y_block)


def count_lines(path):
    with open(path) as f:
        return sum([1 for _ in f])


def load_data(concept_vocabulary, relation_vocabulary, path, debug=False,
              delimiter=' '):
    n_lines = min(10000, count_lines(path)) if debug else count_lines(path)
    bar = progressbar.ProgressBar()
    hs, rs, ts, ys = ([], [], [], [])
    print('loading...: %s' % path)
    with open(path) as f:
        for line in bar(f, max_value=n_lines):
            r, h, t, y = line.strip().split('\t')
            hs.append(
                numpy.array(
                    [concept_vocabulary.get(w.lower(), UNK)
                     for w in h.split(' ')], 'i'
                )
            )
            rs.append(
                numpy.array(
                    [relation_vocabulary.get(w.lower(), UNK)
                     for w in r.split(' ')], 'i'
                )
            )
            ts.append(
                numpy.array(
                    [concept_vocabulary.get(w.lower(), UNK)
                     for w in t.split(' ')], 'i'
                )
            )
            ys.append(
                numpy.array(
                    [float(y)], 'i'  # only used for validation and testing
                )
            )
            if len(hs) == n_lines:
                break
    return (hs, rs, ts, ys)


def load_vocabulary(path):
    with open(path) as f:
        # +2 for UNK and EOS
        word_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    word_ids['<UNK>'] = UNK
    word_ids['<EOS>'] = EOS
    return word_ids


def calculate_unknown_ratio(data):
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total


def load_embedding(path, vocab):
    print('loading...: %s' % path)
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    embedding = numpy.zeros((len(vocab), model.vector_size), 'f')
    unk_indexes = []
    bar = progressbar.ProgressBar()
    for k, v in bar(vocab.items(), max_value=len(vocab)):
        if k in model.vocab:
            embedding[v] = model.word_vec(k)
        else:
            unk_indexes.append(v)
    unk_vector = numpy.sum(embedding, axis=0) / (len(vocab) - len(unk_indexes))
    embedding[unk_indexes] = unk_vector

    unk_ratio = len(unk_indexes) / len(vocab)
    print('Pretrained word embedding covers %.2f%% of vocabulary' % unk_ratio)
    return embedding
