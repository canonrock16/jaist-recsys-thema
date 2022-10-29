from typing import Dict, List, Optional, Text

import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs
from tensorflow_recommenders import layers
from tensorflow_recommenders import metrics as tfrs_metrics
from tensorflow_recommenders.tasks import base


class MyRetrieval(tfrs.tasks.Retrieval):
    def call(
        self,
        query_embeddings: tf.Tensor,
        candidate_embeddings: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
        candidate_sampling_probability: Optional[tf.Tensor] = None,
        candidate_ids: Optional[tf.Tensor] = None,
        compute_metrics: bool = True,
        compute_batch_metrics: bool = True,
        item_weights=None,
    ) -> tf.Tensor:

        scores = tf.linalg.matmul(query_embeddings, candidate_embeddings, transpose_b=True)

        num_queries = tf.shape(scores)[0]
        num_candidates = tf.shape(scores)[1]

        labels = tf.eye(num_queries, num_candidates)

        update_ops = []
        if self._factorized_metrics is not None and compute_metrics:
            update_ops.append(
                self._factorized_metrics.update_state(
                    #
                    # query_embeddings, candidate_embeddings[: tf.shape(query_embeddings)[0]], true_candidate_ids=candidate_ids
                    labels, scores,
                )
            )
        if compute_batch_metrics:
            for metric in self._batch_metrics:
                update_ops.append(metric.update_state(labels, scores))


        if item_weights is not None:
            # tf.print('scores1',scores)
            # tf.print('item_weights',item_weights)
            scores = scores * item_weights
            # tf.print('scores2',scores)

        if self._temperature is not None:
            scores = scores / self._temperature

        if candidate_sampling_probability is not None:
            scores = layers.loss.SamplingProbablityCorrection()(scores, candidate_sampling_probability)

        if self._remove_accidental_hits:
            if candidate_ids is None:
                raise ValueError("When accidental hit removal is enabled, candidate ids " "must be supplied.")
                scores = layers.loss.RemoveAccidentalHits()(labels, scores, candidate_ids)

        
        if self._num_hard_negatives is not None:
            scores, labels = layers.loss.HardNegativeMining(self._num_hard_negatives)(scores, labels)

        if False:
            # scoreを適当にいじる
            item_weight = tf.random.uniform(shape=[num_queries, num_candidates], minval=2, maxval=4)
            fixed_scores = scores * item_weight

            # 対角成分だけオリジナルのスコアで、残りは0の行列
            diag_scores = labels * scores
            # tf.print('diag_scores',diag_scores)
            # 改変後スコアの対角成分を0に直し、オリジナルスコアに置き換える
            fixed_scores = tf.linalg.set_diag(fixed_scores, tf.zeros(num_queries))
            # tf.print('fixed_scores',fixed_scores)
            fixed_scores = fixed_scores + diag_scores
            scores = fixed_scores
            # tf.print('scores',scores)

        
        loss = self._loss(y_true=labels, y_pred=scores, sample_weight=sample_weight)

        for metric in self._loss_metrics:
            update_ops.append(metric.update_state(loss, sample_weight=sample_weight))

        
        with tf.control_dependencies(update_ops):
            return tf.identity(loss)


class RetrievalModel(tfrs.Model):
    def __init__(self, unique_item_ids, unique_user_ids, user_dict_key, item_dict_key, embedding_dimension, metrics_candidate_dataset, loss,num_hard_negatives):

        super().__init__()
        self.user_model = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None, num_oov_indices=0),
                tf.keras.layers.Embedding(len(unique_user_ids), embedding_dimension),
            ]
        )
        self.item_model = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=unique_item_ids, mask_token=None, num_oov_indices=0),
                tf.keras.layers.Embedding(len(unique_item_ids), embedding_dimension),
            ]
        )
        self.task = MyRetrieval(
            loss=loss,
            remove_accidental_hits=True,
            num_hard_negatives=num_hard_negatives,
            metrics=tfr.keras.metrics.MRRMetric(),
            # metrics=tfrs.metrics.FactorizedTopK(candidates=metrics_candidate_dataset.batch(2000).map(self.item_model)),
            batch_metrics=[
                tf.keras.metrics.AUC(num_thresholds=200, curve="PR", summation_method="interpolation", name="auc_metric", from_logits=True)
            ],
        )
        # self.task = tfrs.tasks.Retrieval(metrics=None)
        # self.task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(candidates=metrics_candidate_dataset.batch(128).map(self.item_model)))
        # self.task = tfrs.tasks.Retrieval(metrics=tfr.keras.metrics.MRRMetric())
        self.user_dict_key = user_dict_key
        self.item_dict_key = item_dict_key

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # def compute_loss(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        user_id_list = features[self.user_dict_key]
        item_id_list = features[self.item_dict_key]
        user_embeddings = self.user_model(user_id_list)
        item_embeddings = self.item_model(item_id_list)

        candidate_ids = tf.cast(self.item_model.layers[0](item_id_list), tf.int32)

        if "item_weights" in features.keys():
            item_weights = features["item_weights"]
            return self.task(user_embeddings, item_embeddings, candidate_ids=candidate_ids, item_weights=item_weights)
            # return self.task(user_embeddings, item_embeddings, candidate_ids=candidate_ids, item_weights=item_weights, compute_metrics=not training)

        return self.task(user_embeddings, item_embeddings, candidate_ids=candidate_ids)
        # return self.task(user_embeddings, item_embeddings, candidate_ids=candidate_ids, compute_metrics=not training)
