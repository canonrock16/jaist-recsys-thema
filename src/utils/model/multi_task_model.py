from typing import Dict, Text

import tensorflow as tf
import tensorflow_recommenders as tfrs


class MultiTaskModel(tfrs.models.Model):
    def __init__(
        self,
        unique_item_ids,
        unique_user_ids,
        embedding_dimension,
        rating_weight: float,
        retrieval_weight: float,
        user_dict_key,
        item_dict_key,
        rating_dict_key,
    ) -> None:
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
        self.rating_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

        self.retrieval_task = tfrs.tasks.Retrieval(metrics=None)

        self.user_dict_key = user_dict_key
        self.item_dict_key = item_dict_key
        self.rating_dict_key = rating_dict_key
        # The loss weights.
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        user_embeddings = self.user_model(features[self.user_dict_key])
        item_embeddings = self.item_model(features[self.item_dict_key])

        return (
            user_embeddings,
            item_embeddings,
            self.rating_model(tf.concat([user_embeddings, item_embeddings], axis=1)),
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

        ratings = features.pop(self.rating_dict_key)
        user_embeddings, item_embeddings, rating_predictions = self(features)

        rating_loss = self.rating_task(labels=ratings, predictions=rating_predictions, compute_metrics=not training)
        retrieval_loss = self.retrieval_task(user_embeddings, item_embeddings, compute_metrics=not training)

        return self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss
