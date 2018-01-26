"""Train a CKBC model."""
import argparse

import six

import chainer
from chainer import training
from chainer.training import extensions

from net import BilinearAVG
from utils import load_vocabulary
from utils import load_data
from utils import calculate_unknown_ratio
from utils import fact_pad_concat_convert
from utils import load_embedding
from metrics import CalculateAccuracy


def main():
    parser = argparse.ArgumentParser(description='CKBC')
    parser.add_argument('TRAIN', help='training dataset')
    parser.add_argument('CONCEPT_VOCAB', help='concept vocabulary')
    parser.add_argument('RELATION_VOCAB', help='relation vocabulary')
    parser.add_argument('--validation1',
                        help='validation dataset (1)')
    parser.add_argument('--validation2',
                        help='validation dataset (2)')
    parser.add_argument('--test',
                        help='test dataset')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--concept-unit', type=int, default=256,
                        help='number of concept units')
    parser.add_argument('--relation-unit', type=int, default=256,
                        help='number of relation units')
    parser.add_argument('--dropout', type=int, default=0.1,
                        help='number of layers')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='number of iteration to show log')
    parser.add_argument('--embedding', default='',
                        help='path to pretrained word embedding')
    parser.add_argument('--finetune-embedding', action='store_true',
                        help='finetune pretrained embedding')
    parser.add_argument('--validation-interval', type=int, default=4000,
                        help='number of iteration to evlauate the model '
                        'with validation dataset')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    parser.add_argument('--debug', action='store_true',
                        help='use a small part of training data')
    args = parser.parse_args()

    concept_ids = load_vocabulary(args.CONCEPT_VOCAB)
    relation_ids = load_vocabulary(args.RELATION_VOCAB)
    train_facts = load_data(
        concept_ids,
        relation_ids,
        args.TRAIN,
        debug=args.debug
    )
    train_data = [(h, r, t, y) for h, r, t, y in six.moves.zip(*train_facts)]

    train_head_unk = calculate_unknown_ratio(
        [h for h, _, _, _ in train_data]
    )
    train_relation_unk = calculate_unknown_ratio(
        [r for _, r, _, _ in train_data]
    )
    train_tail_unk = calculate_unknown_ratio(
        [t for _, _, t, _ in train_data]
    )

    embedding = load_embedding(args.embedding, concept_ids) \
        if args.embedding else None
    n_embed = embedding.shape[1] \
        if embedding is not None else args.concept_unit

    print('Concept vocabulary size: %d' % len(concept_ids))
    print('Relation vocabulary size: %d' % len(relation_ids))
    print('Train data size: %d' % len(train_data))
    print('Train head unknown: %.2f' % train_head_unk)
    print('Train relation unknown: %.2f' % train_relation_unk)
    print('Train tail unknown: %.2f' % train_tail_unk)
    if args.embedding:
        print('Pretrained word embedding: %s' % args.embedding)
        print('Fine-tune word embedding: %s' % args.finetune_embedding)

    model = BilinearAVG(
        len(concept_ids),
        len(relation_ids),
        n_embed,
        args.relation_unit,
        args.dropout,
        embedding=embedding
    )
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    if args.embedding != '' and not args.finetune_embedding:
        print('Freezing word embeddings...')
        model.concept_encoder.disable_update()

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    updater = training.StandardUpdater(
        train_iter,
        optimizer,
        converter=fact_pad_concat_convert,
        device=args.gpu
    )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(
        extensions.LogReport(trigger=(args.log_interval, 'iteration'))
    )
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
             'validation/main/accuracy', 'validation/main/threshold',
             'elapsed_time']
        ),
        trigger=(args.log_interval, 'iteration')
    )

    if args.validation1 and args.validation2:
        test_facts = load_data(
            concept_ids,
            relation_ids,
            args.validation1
        )
        test_data1 = [(h, r, t, y)
                      for h, r, t, y in six.moves.zip(*test_facts)]
        test_head_unk = calculate_unknown_ratio(
            [h for h, _, _, _ in test_data1]
        )
        test_relation_unk = calculate_unknown_ratio(
            [r for _, r, _, _ in test_data1]
        )
        test_tail_unk = calculate_unknown_ratio(
            [t for _, _, t, _ in test_data1]
        )
        print('Validation data: %d' % len(test_data1))
        print('Validation head unknown: %.2f' % test_head_unk)
        print('Validation relation unknown: %.2f' % test_relation_unk)
        print('Validation tail unknown: %.2f' % test_tail_unk)

        test_facts = load_data(
            concept_ids,
            relation_ids,
            args.validation2
        )
        test_data2 = [(h, r, t, y)
                      for h, r, t, y in six.moves.zip(*test_facts)]
        test_head_unk = calculate_unknown_ratio(
            [h for h, _, _, _ in test_data2]
        )
        test_relation_unk = calculate_unknown_ratio(
            [r for _, r, _, _ in test_data2]
        )
        test_tail_unk = calculate_unknown_ratio(
            [t for _, _, t, _ in test_data2]
        )
        print('Validation data: %d' % len(test_data2))
        print('Validation head unknown: %.2f' % test_head_unk)
        print('Validation relation unknown: %.2f' % test_relation_unk)
        print('Validation tail unknown: %.2f' % test_tail_unk)

        trainer.extend(
            CalculateAccuracy(
                model, test_data1, test_data2, device=args.gpu,
                key_accuracy='validation/main/accuracy',
                key_threshold='validation/main/threshold'
            ),
            trigger=(args.validation_interval, 'iteration')
        )

    print('start training')
    trainer.run()


if __name__ == '__main__':
    main()
