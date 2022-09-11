import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from tqdm import tqdm
import sys

from ..utils.model.retrieval_model import RetrievalModel

batch_size = 200
embedding_dimension = 512
learning_rate = 0.1
early_stopping_flg = True
tensorboard_flg = False
log_path = "./logs/MIND/"
max_epoch_num = 20


def make_click_df(behaviors_df):
    user2clicks = {}
    for index, data in tqdm(behaviors_df.iterrows()):
        user = data["User_ID"]
        impressions = data["Impressions"].split(" ")
        clicks = []
        for impression in impressions:
            if impression[-1] == "1":
                clicks.append(impression[:-2])
        if user not in user2clicks:
            user2clicks[user] = clicks
        else:
            user2clicks[user] = user2clicks[user] + clicks

    user_list = []
    click_list = []
    for user, v in tqdm(user2clicks.items()):
        for click in v:
            user_list.append(user)
            click_list.append(click)

    print("user_list", len(user_list))
    print("click_list", len(click_list))

    click_df = pd.DataFrame(list(zip(user_list, click_list)), columns=["user_id", "item_id"])

    return click_df


def main():
    train_behaviors_df = pd.read_table(
        "data/MIND/MINDsmall_train/behaviors.tsv", names=("Impression_ID", "User_ID", "Time", "History", "Impressions")
    )
    val_behaviors_df = pd.read_table("data/MIND/MINDsmall_dev/behaviors.tsv", names=("Impression_ID", "User_ID", "Time", "History", "Impressions"))
    test_behaviors_df = pd.read_table("data/MIND/MINDsmall_dev/behaviors.tsv", names=("Impression_ID", "User_ID", "Time", "History", "Impressions"))
    print("unique user number of train", len(train_behaviors_df["User_ID"].unique()))
    print("unique user number of val", len(val_behaviors_df["User_ID"].unique()))
    print("unique user number of test", len(test_behaviors_df["User_ID"].unique()))

    train_click_df = make_click_df(train_behaviors_df)
    train_ratings = tf.data.Dataset.from_tensor_slices({"user_id": train_click_df["user_id"], "item_id": train_click_df["item_id"]})
    val_click_df = make_click_df(val_behaviors_df)
    val_ratings = tf.data.Dataset.from_tensor_slices({"user_id": val_click_df["user_id"], "item_id": val_click_df["item_id"]})
    test_click_df = make_click_df(test_behaviors_df)
    test_ratings = tf.data.Dataset.from_tensor_slices({"user_id": test_click_df["user_id"], "item_id": test_click_df["item_id"]})

    train = train_ratings.batch(batch_size)
    val = val_ratings.batch(batch_size)
    test = test_ratings.batch(batch_size)

    unique_user_ids = np.array(
        list((set(train_click_df["user_id"].unique()) | set(val_click_df["user_id"].unique()) | set(test_click_df["user_id"].unique())))
    )
    unique_item_ids = np.array(
        list(set(train_click_df["item_id"].unique()) | set(val_click_df["item_id"].unique()) | set(test_click_df["item_id"].unique()))
    )
    unique_item_dataset = tf.data.Dataset.from_tensor_slices(unique_item_ids)

    print(len(unique_user_ids))
    print(len(unique_item_ids))
    print(len(set(train_click_df["item_id"].unique())))

    strategy = tf.distribute.MirroredStrategy()
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
