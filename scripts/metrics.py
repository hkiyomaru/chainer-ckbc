"""Metrics to evaluate KBC models."""
import chainer

from utils import fact_pad_concat_convert


class CalculateAccuracy(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, model, test_data, key,
                 batch_size=100, device=-1):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch_size = batch_size
        self.device = device

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch_size):
                hs, rs, ts, ys = fact_pad_concat_convert(
                    self.test_data[i:i + self.batch_size],
                    device=self.device,
                    test=True
                )
                references.extend(ys.flatten().tolist())
                hypotheses.extend(
                    (self.model.predict(hs, rs, ts).data > 0.5).tolist()
                )

        acc = sum([r == h for r, h in zip(references, hypotheses)]) \
            / len(references)
        chainer.report({self.key: acc})
