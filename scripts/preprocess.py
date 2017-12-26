"""Preprocess dataset to make concept vocabulary."""
import argparse
import collections
import os
import progressbar

from utils import count_lines


def make_vocab(path, n_vocab, delimiter=' '):
    n_lines = count_lines(args.TRAIN)
    bar = progressbar.ProgressBar()
    words = []
    with open(args.TRAIN) as f:
        for line in bar(f, max_value=n_lines):
            _, h, t, _ = line.strip().split('\t')
            words.extend(h.strip().split(delimiter))
            words.extend(t.strip().split(delimiter))
    counter = collections.Counter(words)
    vocab = [w for w, c in counter.most_common(n_vocab)]
    return vocab


def main(args):
    vocab = make_vocab(args.TRAIN, args.n_vocab)

    dirname = os.path.dirname(args.TRAIN)
    output_path = os.path.join(dirname, 'con.txt')
    with open(output_path, 'w') as f:
        for word in vocab:
            f.write('%s\n' % word)
    print('concept vocabulary has built. (%s)' % output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocessor')
    parser.add_argument('TRAIN',
                        help='path to training dataset')
    parser.add_argument('--n-vocab', type=int, default=30000,
                        help='vocabulary size')
    args = parser.parse_args()
    main(args)
