"""Commonsense knowledge base completion models."""
import chainer
import chainer.functions as F
import chainer.links as L

from utils import PAD


class BilinearCKBC(chainer.Chain):

    def __init__(self, n_concept_vocab, n_relation_vocab, n_concept_units,
                 n_relation_units, n_dropout, encoding='average',
                 embedding=None):

        super(BilinearCKBC, self).__init__()
        with self.init_scope():
            self.concept_encoder = ConceptEncoder(
                n_concept_vocab,
                n_concept_units,
                n_dropout,
                encoding=encoding,
                embedding=embedding
            )
            self.relation_encoder = L.EmbedID(
                n_relation_vocab,
                n_relation_units ** 2
            )
            if encoding == 'average':
                self.l_concept = L.Linear(
                    n_concept_units,
                    n_relation_units
                )
            elif encoding.startswith('bilstm'):
                self.l_concept = L.Linear(
                    n_concept_units * 2,
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

        # transform concept representations
        l_hhs = F.tanh(F.dropout(self.l_concept(hhs), ratio=self.n_dropout))
        l_hts = F.tanh(F.dropout(self.l_concept(hts), ratio=self.n_dropout))

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


class ConceptEncoder(chainer.Chain):

    def __init__(self, n_vocab, n_units, n_dropout=0.0, encoding='average',
                 embedding=None):

        super(ConceptEncoder, self).__init__()
        with self.init_scope():
            self.concept_encoder = L.EmbedID(
                n_vocab,
                n_units,
                initialW=embedding
            )
            self.bilstm = L.NStepBiLSTM(
                1,
                n_units,
                n_units,
                n_dropout
            )

        # encoding should be average, bilstm-max-pooling, or bilstm-last-state.
        self.encoding = encoding.replace('-', '_')
        self.n_units = n_units

    def __call__(self, cs):
        """Calculate the loss between outputs and ys.

        Args:
            cs: The concepts of facts.

        Returns:
            hcs: Vector representation for cs.

        """
        ecs = self.concept_encoder(cs)
        hcs = eval('self.%s' % self.encoding)(ecs, cs)
        return hcs

    def average(self, ecs, cs):
        ecs = ecs * (cs != PAD)[:, :, None]
        hcs = F.average(ecs, axis=1)
        return hcs

    def bilstm_max_pooling(self, ecs, cs):
        hcs = self.bilstm_encode(ecs, cs)
        hcs = F.vstack(
            [F.max(x, axis=0) for x in hcs]
        )
        return hcs

    def bilstm_last_state(self, ecs, cs):
        hcs = self.bilstm_encode(ecs, cs)
        n_units = self.n_units
        hcs = F.vstack(
            [F.hstack((x[0][:n_units], x[-1][n_units:])) for x in hcs]
        )
        return hcs

    def bilstm_encode(self, ecs, cs):
        batch_size, _ = cs.shape
        ecs = F.separate(ecs, axis=0)
        masks = self.xp.vsplit(cs != PAD, batch_size)
        masked_ecs = [ex[mask.reshape((-1, ))] for ex, mask in zip(ecs, masks)]
        _, _, hcs = self.bilstm(None, None, masked_ecs)
        return hcs
