import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import atexit
import tensorflow_recommenders as tfrs
import pickle
import time

from tqdm import tqdm
from src.utils.model.retrieval_model import RetrievalModel

batch_size = 100
embedding_dimension = 256
learning_rate = 0.1
early_stopping_flg = True
tensorboard_flg = False
log_path = "./logs/MIND/"
max_epoch_num = 20
negative_weight = 10
negative_weight_mode = True
num_hard_negatives = 10000
# make_new_dataset = False

train_negative_path = './data/tf_dataset/MIND_large_train_negative_mode/'
train_dataset_path = './data/tf_dataset/MIND_large_train/'
val_dataset_path = './data/tf_dataset/MIND_large_val/'
test_dataset_path = './data/tf_dataset/MIND_large_test/'
unique_item_dataset_path = './data/tf_dataset/MIND_large_unique_item_dataset/'
unique_user_ids_path = "./data/other/unique_user_ids.pickle"
unique_item_ids_path = "./data/other/unique_item_ids.pickle"

print('batch_size',batch_size)
print('embedding_dimension',embedding_dimension)
print('learning_rate',learning_rate)
print('early_stopping_flg',early_stopping_flg)
print('max_epoch_num',max_epoch_num)
print('negative_weight_mode',negative_weight_mode)
print('negative_weight',negative_weight)
print('num_hard_negatives',num_hard_negatives)
# print('make_new_dataset',make_new_dataset)


def convert_df(behaviors_df, flg):
    user2clicks = {}
    for index, data in tqdm(behaviors_df.iterrows()):
        user = data["User_ID"]
        impressions = data["Impressions"].split(" ")
        clicks = []
        for impression in impressions:
            if impression[-1] == flg:
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
    start = time.time()
    # train_behaviors_df = pd.read_table(
    #     "data/MIND/small_train/behaviors.tsv", names=("Impression_ID", "User_ID", "Time", "History", "Impressions")
    # )
    # val_behaviors_df = pd.read_table("data/MIND/small_val/behaviors.tsv", names=("Impression_ID", "User_ID", "Time", "History", "Impressions"))
    # test_behaviors_df = pd.read_table("data/MIND/small_val/behaviors.tsv", names=("Impression_ID", "User_ID", "Time", "History", "Impressions"))

    # train = tf.data.experimental.load(train_dataset_path)
    # val = tf.data.experimental.load(val_dataset_path)
    # test = tf.data.experimental.load(test_dataset_path)
    # unique_item_dataset = tf.data.experimental.load(unique_item_dataset_path)
    # with open(unique_user_ids_path, mode="rb") as f:
    #     unique_user_ids = pickle.load(f)
    # with open(unique_item_ids_path, mode="rb") as f:
    #     unique_item_ids = pickle.load(f)

    # if make_new_dataset:
    train_behaviors_df = pd.read_table(
    "data/MIND/large_train/behaviors.tsv", names=("Impression_ID", "User_ID", "Time", "History", "Impressions")
    )
    val_behaviors_df = pd.read_table("data/MIND/large_val/behaviors.tsv", names=("Impression_ID", "User_ID", "Time", "History", "Impressions"))
    test_behaviors_df = pd.read_table("data/MIND/large_test/behaviors.tsv", names=("Impression_ID", "User_ID", "Time", "History", "Impressions"))

    print("unique user number of train", len(train_behaviors_df["User_ID"].unique()))
    print("unique user number of val", len(val_behaviors_df["User_ID"].unique()))
    print("unique user number of test", len(test_behaviors_df["User_ID"].unique()))

    # 規模縮小
    # if False:
    #     train_behaviors_df = train_behaviors_df[:10000]
    #     val_behaviors_df = val_behaviors_df[:10000]
    #     test_behaviors_df = test_behaviors_df[:10000]
        

    train_click_df = convert_df(train_behaviors_df, "1")
    val_click_df = convert_df(val_behaviors_df, "1")
    test_click_df = convert_df(test_behaviors_df, "1")

    # バッチサイズで割り切れるように丸める
    step_size = int(len(train_click_df) / batch_size)
    train_click_df = train_click_df[: step_size * batch_size]

    train_ratings = tf.data.Dataset.from_tensor_slices({"user_id": train_click_df["user_id"], "item_id": train_click_df["item_id"]})
    val_ratings = tf.data.Dataset.from_tensor_slices({"user_id": val_click_df["user_id"], "item_id": val_click_df["item_id"]})
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

    # tf.data.experimental.save(dataset=train, path=train_dataset_path)
    # tf.data.experimental.save(dataset=val, path=val_dataset_path)
    # tf.data.experimental.save(dataset=test, path=test_dataset_path)
    # tf.data.experimental.save(dataset=unique_item_dataset, path=unique_item_dataset_path)
    # with open(unique_user_ids_path, mode="wb") as f:
    #     pickle.dump(unique_user_ids, f)
    # with open(unique_item_ids_path, mode="wb") as f:
    #     pickle.dump(unique_item_ids, f)



    if negative_weight_mode:
        # train = tf.data.experimental.load(train_negative_path)
        # if make_new_dataset:
        print('negative weight mode')
        train_impression_df = convert_df(train_behaviors_df, "0")
        train_impression_df = (
            train_impression_df.groupby(["user_id", "item_id"]).size().sort_values(ascending=False).reset_index(name="count")
        )

        user_id2seen_items = {}
        for index, data in tqdm(train_impression_df.iterrows(),total=len(train_impression_df)):
            user_id = data["user_id"]
            item_id = data["item_id"]
            count = data["count"]
            
            if user_id not in user_id2seen_items:
                user_id2seen_items[user_id] = [{"item_id":item_id,"count":count}]
            else:
                user_id2seen_items[user_id].append({"item_id":item_id,"count":count})

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
                                weights[j] = seen_item["count"] + negative_weight
                item_weights_by_batch.append(weights)
            item_weights.append(item_weights_by_batch)

        item_weights = np.array(item_weights)
        print("item_weights.shape", item_weights.shape)
        train_ratings = tf.data.Dataset.from_tensor_slices(
            {
                "user_id": train_click_df["user_id"],
                "item_id": train_click_df["item_id"],
                "item_weights": item_weights.reshape([step_size * batch_size, batch_size]),
            }
        )
        train = train_ratings.batch(batch_size)

        # 各種チェック
        # print('start check')
        # indexes = np.where(item_weights == 2)
        # item_weightが2になっているインデックスに相当するユーザーとアイテムが、本当にseen_in_list_dfにあるかどうか検査
        # for i, j, k in zip(indexes[0], indexes[1], indexes[2]):
        #     if i == 0:
        #         for batch in train.take(1):
        #             user_ids = batch["user_id"].numpy()
        #             item_ids = batch["item_id"].numpy()
        #             user_id = user_ids[j].decode("utf-8")
        #             item_id = item_ids[k].decode("utf-8")

        #             result = train_impression_df.query(f'user_id == "{user_id}" and item_id == "{item_id}"')
        #             if len(result) == 0:
        #                 print("zero")
        #             else:
        #                 # print('ok')
        #                 pass
        # # item_weightの対角成分が2になっていないことを確認
        # for i, j, k in zip(indexes[0], indexes[1], indexes[2]):
        #     if j == k:
        #         print("NG")
        # # item_weightsの内容と、trainから出てくる内容が同一であることをチェック
        # for i, batch in enumerate(train.take(1)):
        #     if (item_weights[i] != batch["item_weights"]).numpy().all():
        #         print("NG")

        # tf.data.experimental.save(dataset=train, path=train_negative_path)



    # strategy = tf.distribute.MirroredStrategy()
    # atexit.register(strategy._extended._collective_ops._pool.close) # type: ignore
    # with strategy.scope():
    print('make data time',time.time() - start)
    time2 = time.time()

    model = RetrievalModel(
        unique_user_ids=unique_user_ids,
        unique_item_ids=unique_item_ids,
        user_dict_key="user_id",
        item_dict_key="item_id",
        embedding_dimension=embedding_dimension,
        metrics_candidate_dataset=unique_item_dataset,
        num_hard_negatives=num_hard_negatives,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM),
    )
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate))

    callbacks = []
    if early_stopping_flg:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                # monitor="val_total_loss",
                monitor="val_mrr_metric",
                min_delta=0,
                patience=3,
                verbose=0,
                mode="max",
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

    print('start training')
    model.fit(x=train, validation_data=val, epochs=max_epoch_num, callbacks=callbacks)
    print('training time',time.time() - time2)
    time3 = time.time()

    # model.task.factorized_metrics = tfrs.metrics.FactorizedTopK(
        #   candidates=tfrs.layers.factorized_top_k.BruteForce().index_from_dataset(unique_item_dataset.batch(8192).map(model.item_model)))
    # model.compile()

    print('start evaluate')
    model.evaluate(test, return_dict=True)
    print('evaluate time',time.time() - time3)


    print('whole time',time.time() - start)


if __name__ == "__main__":
    main()
