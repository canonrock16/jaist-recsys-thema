import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import atexit
import tensorflow_recommenders as tfrs

from sklearn.model_selection import train_test_split
from src.utils.model.retrieval_model import RetrievalModel
from tqdm import tqdm

val_rate = 0.2
test_rate = 0.1
batch_size = 200
embedding_dimension = 1024
learning_rate = 0.1
early_stopping_flg = True
tensorboard_flg = False
log_path = "./logs/MIND/"
max_epoch_num = 20

print("val_rate", val_rate)
print("test_rate", test_rate)
print("batch_size", batch_size)
print("embedding_dimension", embedding_dimension)
print("learning_rate", learning_rate)
print("early_stopping_flg", early_stopping_flg)
print("max_epoch_num", max_epoch_num)


def main():
    behaviors_df = pd.read_csv("data/RPCD/user_activity.csv", names=("item_id", "user_id", "event_type", "create_timestamp"))

    seen_df = behaviors_df.query('event_type == "seen"')
    count_df = pd.DataFrame(seen_df["user_id"].value_counts()).reset_index().rename(columns={"index": "user_id", "user_id": "count"})
    unique_user_ids = list(count_df.query("count >= 10")["user_id"])
    seen_df = seen_df[seen_df["user_id"].isin(unique_user_ids)]
    seen_in_list_df = (
        behaviors_df.query('event_type == "seen_in_list"')
        .groupby(["user_id", "item_id"])
        .size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )

    train_val_df, test_df = train_test_split(seen_df, test_size=test_rate, stratify=seen_df["user_id"], random_state=1)
    train_df, val_df = train_test_split(train_val_df, test_size=val_rate, stratify=train_val_df["user_id"], random_state=1)

    # バッチサイズで割り切れるように丸める
    step_size = int(len(train_df) / batch_size)
    train_df = train_df[: step_size * batch_size]

    print("train_df unique user_id num", len(train_df["user_id"].unique()))
    print("val_df unique user_id num", len(val_df["user_id"].unique()))
    print("test_df unique user_id num", len(test_df["user_id"].unique()))

    train_ratings = tf.data.Dataset.from_tensor_slices({"user_id": train_df["user_id"], "item_id": train_df["item_id"]})
    val_ratings = tf.data.Dataset.from_tensor_slices({"user_id": val_df["user_id"], "item_id": val_df["item_id"]})
    test_ratings = tf.data.Dataset.from_tensor_slices({"user_id": test_df["user_id"], "item_id": test_df["item_id"]})

    train = train_ratings.batch(batch_size)
    val = val_ratings.batch(batch_size)
    test = test_ratings.batch(batch_size)

    unique_user_ids = np.array(list((set(train_df["user_id"].unique()) | set(val_df["user_id"].unique()) | set(test_df["user_id"].unique()))))
    unique_item_ids = np.array(list(set(train_df["item_id"].unique()) | set(val_df["item_id"].unique()) | set(test_df["item_id"].unique())))
    unique_item_dataset = tf.data.Dataset.from_tensor_slices(unique_item_ids)

    if True:
        user_id2seen_items = {}
        seen_user_ids = list(seen_in_list_df["user_id"].unique())
        for seen_user_id in tqdm(seen_user_ids):
            user_id2seen_items[seen_user_id] = []
            seen_items = seen_in_list_df.query(f'user_id == "{seen_user_id}"')
            for i, item in seen_items.iterrows():
                user_id2seen_items[seen_user_id].append({"item_id": item["item_id"], "count": item["count"]})

        item_weights = []
        for batch in tqdm(train):
            # そのバッチに含まれるuser_idとitem_id達
            user_ids = batch["user_id"].numpy()
            item_ids = batch["item_id"].numpy()

            item_weights_by_batch = []
            for i, user_id in enumerate(user_ids):
                # 基本のweightsは1にする
                weights = np.ones(len(item_ids), dtype="float32")
                decoded_user_id = user_id.decode("utf-8")
                # もしそのユーザーのviewログがあるアイテムがあり、かつそのアイテムがバッチの中に存在して、ユーザーがクリックしていなかったら、weightを上げる
                if decoded_user_id in user_id2seen_items:
                    seen_items = user_id2seen_items[decoded_user_id]
                    # 各seen_itemがバッチの中に存在するか？
                    for seen_item in seen_items:
                        for j, item_id in enumerate(item_ids):
                            decoded_item_id = item_id.decode("utf-8")
                            if seen_item["item_id"] == decoded_item_id and i != j:
                                weights[j] = seen_item["count"] + 1
                                # weights[j] = seen_item["count"]+10
                item_weights_by_batch.append(weights)
            item_weights.append(item_weights_by_batch)

        item_weights = np.array(item_weights)
        print("item_weights.shape", item_weights.shape)
        train_ratings = tf.data.Dataset.from_tensor_slices(
            {
                "user_id": train_df["user_id"],
                "item_id": train_df["item_id"],
                "item_weights": item_weights.reshape([step_size * batch_size, batch_size]),
            }
        )
        train = train_ratings.batch(batch_size)

        # 各種チェック
        indexes = np.where(item_weights == 2)
        # item_weightが2になっているインデックスに相当するユーザーとアイテムが、本当にseen_in_list_dfにあるかどうか検査
        for i, j, k in zip(indexes[0], indexes[1], indexes[2]):
            if i == 0:
                for batch in train.take(1):
                    user_ids = batch["user_id"].numpy()
                    item_ids = batch["item_id"].numpy()
                    user_id = user_ids[j].decode("utf-8")
                    item_id = item_ids[k].decode("utf-8")

                    result = seen_in_list_df.query(f'user_id == "{user_id}" and item_id == "{item_id}"')
                    if len(result) == 0:
                        print("zero")
                    else:
                        # print('ok')
                        pass
        # item_weightの対角成分が2になっていないことを確認
        for i, j, k in zip(indexes[0], indexes[1], indexes[2]):
            if j == k:
                print("NG")
        # item_weightsの内容と、trainから出てくる内容が同一であることをチェック
        for i, batch in enumerate(train):
            if (item_weights[i] != batch["item_weights"]).numpy().all():
                print("NG")

    strategy = tf.distribute.MirroredStrategy()
    atexit.register(strategy._extended._collective_ops._pool.close)  # type: ignore
    with strategy.scope():
        model = RetrievalModel(
            unique_user_ids=unique_user_ids,
            unique_item_ids=unique_item_ids,
            user_dict_key="user_id",
            item_dict_key="item_id",
            embedding_dimension=embedding_dimension,
            metrics_candidate_dataset=unique_item_dataset,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM),
        )
        model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate))

    callbacks = []
    if early_stopping_flg:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_total_loss",
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

    model.task.factorized_metrics = tfrs.metrics.FactorizedTopK(
        candidates=tfrs.layers.factorized_top_k.BruteForce().index_from_dataset(unique_item_dataset.batch(8192).map(model.item_model))
    )
    model.compile()
    model.evaluate(test, return_dict=True)


if __name__ == "__main__":
    main()
