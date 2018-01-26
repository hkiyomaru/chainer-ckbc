"""Metrics to evaluate KBC models."""
from collections import defaultdict

import chainer

from utils import fact_pad_concat_convert


class CalculateAccuracy(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, model, devel_data1, devel_data2, key_accuracy,
                 key_threshold, batch_size=100, device=-1):
        self.model = model
        self.devel_data1 = devel_data1
        self.devel_data2 = devel_data2
        self.key_accuracy = key_accuracy
        self.key_threshold = key_threshold
        self.batch_size = batch_size
        self.device = device

    def __call__(self, trainer):
        # Step 1: Determines a threshold to classify facts
        with chainer.no_backprop_mode():
            references = []
            hypotheses = defaultdict(list)
            for i in range(0, len(self.devel_data1), self.batch_size):
                hs, rs, ts, ys = fact_pad_concat_convert(
                    self.devel_data1[i:i + self.batch_size],
                    device=self.device,
                    test=True
                )
                references.extend(ys.flatten().tolist())
                for scaled_threshold in range(11):
                    threshold = scaled_threshold / 10
                    hypotheses[threshold].extend(
                        (self.model.predict(hs, rs, ts).data > threshold)
                        .tolist()
                    )
        best_threshold, _ = sorted(
            {key: sum([h == r
                       for r, h
                       in zip(hypotheses[key], references)])
             for key in hypotheses.keys()}.items(),
            key=lambda x: x[1],
            reverse=True
        )[0]
        chainer.report({self.key_threshold: best_threshold})

        # Step 2: Calculates accuracy based on the threshold
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.devel_data2), self.batch_size):
                hs, rs, ts, ys = fact_pad_concat_convert(
                    self.devel_data2[i:i + self.batch_size],
                    device=self.device,
                    test=True
                )
                references.extend(ys.flatten().tolist())
                hypotheses.extend(
                    (self.model.predict(hs, rs, ts).data > best_threshold)
                    .tolist()
                )
        acc = sum([r == h for r, h in zip(references, hypotheses)]) \
            / len(references)
        chainer.report({self.key_accuracy: acc})
