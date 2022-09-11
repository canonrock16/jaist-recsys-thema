from typing import Dict, Text

import tensorflow as tf
import tensorflow_recommenders as tfrs


class RetrievalModel(tfrs.Model):
    def __init__(self, unique_item_ids, unique_user_ids, user_dict_key, item_dict_key, embedding_dimension,metrics_candidate_dataset):
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
        # self.task = tfrs.tasks.Retrieval(metrics=None)
        self.task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(candidates=metrics_candidate_dataset.batch(128).map(self.item_model)))
        self.user_dict_key = user_dict_key
        self.item_dict_key = item_dict_key

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_model(features[self.user_dict_key])
        item_embeddings = self.item_model(features[self.item_dict_key])

        return self.task(user_embeddings, item_embeddings, compute_metrics=not training)
