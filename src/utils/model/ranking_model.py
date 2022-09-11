from typing import Dict, Text

import tensorflow as tf
import tensorflow_recommenders as tfrs


class RankingModel(tfrs.Model):
    def __init__(self, unique_item_ids, unique_user_ids, embedding_dimension, user_dict_key, item_dict_key, rating_dict_key):
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

        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        self.user_dict_key = user_dict_key
        self.item_dict_key = item_dict_key
        self.rating_dict_key = rating_dict_key

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        user_embeddings = self.user_model(features[self.user_dict_key])
        item_embeddings = self.item_model(features[self.item_dict_key])

        return (
            user_embeddings,
            item_embeddings,
            self.rating_model(tf.concat([user_embeddings, item_embeddings], axis=1)),
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

        labels = features.pop(self.rating_dict_key)
        _, _, predictions = self(features)

        return self.task(labels=labels, predictions=predictions, compute_metrics=not training)
