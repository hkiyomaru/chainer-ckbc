"""Commonsense knowledge base completion models."""
import chainer
import chainer.functions as F
import chainer.links as L

from utils import PAD


class BilinearCKBC(chainer.Chain):

    def __init__(self, n_concept_vocab, n_relation_vocab, n_concept_units,
                 n_relation_units, n_dropout, embedding=None):

        super(BilinearCKBC, self).__init__()
        with self.init_scope():
            self.concept_encoder = L.EmbedID(
                n_concept_vocab,
                n_concept_units,
                initialW=embedding
            )
            self.relation_encoder = L.EmbedID(
                n_relation_vocab,
                n_relation_units ** 2
            )
            self.l_concept = L.Linear(
                n_concept_units,
                n_relation_units
            )

        self.n_relation_units = n_relation_units
        self.n_dropout = n_dropout

    def __call__(self, hs, rs, ts, ys):
        """Calculate the loss between outputs and ys.

        Args:
            hs: The heads of facts.
            rs: The relations of facts.
            ts: The tails of facts.
            ys: The labels which indicate whether the facts are correct.

        Returns:
            loss: The cross-entropy loss for outputs and ys.

        """
        batch_size, max_length_h = hs.shape
        _, max_length_t = ts.shape

        hhs = self.concept_encoder(hs)
        hts = self.concept_encoder(ts)
        hrs = self.relation_encoder(rs)

        # embedding vectors which corresponds to PAD should be zeros
        hhs = hhs * (hs != PAD)[:, :, None]
        hts = hts * (ts != PAD)[:, :, None]

        # calculate average over embeddings
        hhs = F.average(hhs, axis=1)
        hts = F.average(hts, axis=1)

        # transform concept representations
        l_hhs = F.tanh(
            F.dropout(self.l_concept(hhs), ratio=self.n_dropout)
        )
        l_hts = F.tanh(
            F.dropout(self.l_concept(hts), ratio=self.n_dropout)
        )

        # reshape hrs
        hrs = F.reshape(
            hrs,
            (batch_size, self.n_relation_units, self.n_relation_units)
        )

        # calculate bilinear outputs
        outputs = F.flatten(
            F.batch_matmul(F.batch_matmul(l_hhs, hrs, transa=True), l_hts)
        )

        loss = F.sigmoid_cross_entropy(outputs, ys)
        chainer.report({'loss': loss.data}, self)
        return loss

    def predict(self, hs, rs, ts):
        batch_size, max_length_h = hs.shape
        _, max_length_t = ts.shape

        hhs = self.concept_encoder(hs)
        hts = self.concept_encoder(ts)
        hrs = self.relation_encoder(rs)

        # embedding vectors which corresponds to PAD should be zeros
        hhs = hhs * (hs != PAD)[:, :, None]
        hts = hts * (ts != PAD)[:, :, None]

        # calculate average over embeddings
        hhs = F.average(hhs, axis=1)
        hts = F.average(hts, axis=1)

        # transform concept representations
        l_hhs = F.tanh(
            F.dropout(self.l_concept(hhs), ratio=self.n_dropout)
        )
        l_hts = F.tanh(
            F.dropout(self.l_concept(hts), ratio=self.n_dropout)
        )

        # reshape hrs
        hrs = F.reshape(
            hrs,
            (batch_size, self.n_relation_units, self.n_relation_units)
        )

        # calculate bilinear outputs
        outputs = F.flatten(
            F.sigmoid(
                F.batch_matmul(F.batch_matmul(l_hhs, hrs, transa=True), l_hts)
            )
        )
        return outputs
