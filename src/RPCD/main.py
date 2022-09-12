import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import atexit
from sklearn.model_selection import train_test_split

from src.utils.model.retrieval_model import RetrievalModel

val_rate = 0.2
test_rate = 0.1
batch_size = 200
embedding_dimension = 512
learning_rate = 0.1
early_stopping_flg = True
tensorboard_flg = False
log_path = "./logs/MIND/"
max_epoch_num = 20


def main():
    behaviors_df = pd.read_csv("data/RentalProperties/user_activity.csv", names=("item_id", "user_id", "event_type", "create_timestamp"))

    seen_df = behaviors_df.query('event_type == "seen"')
    count_df = pd.DataFrame(seen_df["user_id"].value_counts()).reset_index().rename(columns={"index": "user_id", "user_id": "count"})
    unique_user_ids = list(count_df.query("count >= 10")["user_id"])
    seen_df = seen_df[seen_df["user_id"].isin(unique_user_ids)]

    train_val_df, test_df = train_test_split(seen_df, test_size=test_rate, stratify=seen_df["user_id"])
    train_df, val_df = train_test_split(train_val_df, test_size=val_rate, stratify=train_val_df["user_id"])

    print(len(train_df["user_id"].unique()))
    print(len(val_df["user_id"].unique()))
    print(len(test_df["user_id"].unique()))

    train_ratings = tf.data.Dataset.from_tensor_slices({"user_id": train_df["user_id"], "item_id": train_df["item_id"]})
    val_ratings = tf.data.Dataset.from_tensor_slices({"user_id": val_df["user_id"], "item_id": val_df["item_id"]})
    test_ratings = tf.data.Dataset.from_tensor_slices({"user_id": test_df["user_id"], "item_id": test_df["item_id"]})

    train = train_ratings.batch(batch_size)
    val = val_ratings.batch(batch_size)
    test = test_ratings.batch(batch_size)

    unique_user_ids = np.array(list((set(train_df["user_id"].unique()) | set(val_df["user_id"].unique()) | set(test_df["user_id"].unique()))))
    unique_item_ids = np.array(list(set(train_df["item_id"].unique()) | set(val_df["item_id"].unique()) | set(test_df["item_id"].unique())))
    unique_item_dataset = tf.data.Dataset.from_tensor_slices(unique_item_ids)

    strategy = tf.distribute.MirroredStrategy()
    atexit.register(strategy._extended._collective_ops._pool.close) # type: ignore
    with strategy.scope():
        model = RetrievalModel(
            unique_user_ids=unique_user_ids,
            unique_item_ids=unique_item_ids,
            user_dict_key="user_id",
            item_dict_key="item_id",
            embedding_dimension=embedding_dimension,
            metrics_candidate_dataset=unique_item_dataset,
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))

    callbacks = []
    if early_stopping_flg:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="total_loss",
                min_delta=0,
                patience=3,
                verbose=0,
                mode="auto",
                baseline=None,
                restore_best_weights=False,
            )
        )
    if tensorboard_flg:
        tfb_log_path = log_path + datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=tfb_log_path,
                histogram_freq=1,
            )
        )

    model.fit(x=train, validation_data=val, epochs=max_epoch_num, callbacks=callbacks)
    model.evaluate(test, return_dict=True)


if __name__ == "__main__":
    main()
