import pandas as pd
import datetime
from ping3 import ping
import random
import json
import re
import multiprocessing as mp


def gen_tx(tx_list, url_list, user_list, user_url_list):
    send_ts = datetime.datetime.now()

    # user's latency
    from_user_idx = random.randint(0, len(user_list) - 1)
    url = user_url_list[from_user_idx]
    latency = ping(url)
    while not latency:
        url = random.choice(url_list)
        latency = ping(url)

    # fluctuation
    fluc = ping(random.choice(url_list))
    while not fluc:
        fluc = ping(random.choice(url_list))

    latency = 0.7 * latency + 0.3 * fluc

    recv_ts = send_ts + datetime.timedelta(seconds=latency)

    from_user = user_list[from_user_idx]
    to_user = random.choice(user_list)
    value = random.random() * 10 # !转账额度怎么生成

    label = 0
    tx_list.append([send_ts, recv_ts, latency, label, from_user, to_user, value])


if __name__ == "__main__":
    with open("./website.json", "r", encoding="utf8") as f:
        website_list = json.load(f)

    pattern = re.compile("http://(.+?)/")
    url_list = [re.findall(pattern, dic["home"])[0] for dic in website_list]

    df_user = pd.read_csv("./user.csv")
    user_list = df_user["address"].values
    user_url_list = df_user["url"].values

    tx_list = mp.Manager().list()
    n = 100000
    num_jobs_running = 500
    for i in range(n // num_jobs_running):
        jobs = []
        for _ in range(num_jobs_running):
            proc = mp.Process(
                target=gen_tx, args=(tx_list, url_list, user_list, user_url_list)
            )
            jobs.append(proc)

        for j in jobs:
            j.start()
        for j in jobs:
            j.join()

        print(f"Finished round {i+1}")

    print(len(tx_list))
    print(tx_list[:5])

    tx_list_sorted = sorted(tx_list, key=lambda tx: tx[1])
    for i in range(1, len(tx_list_sorted)):
        if tx_list_sorted[i][0] < tx_list_sorted[i - 1][0]:
            tx_list_sorted[i][3] = 1

    print("------")
    print(tx_list_sorted[:5])

    df_tx = pd.DataFrame(
        columns=[
            "send_timestamp",
            "recv_timestamp",
            "latency",
            "label",
            "from",
            "to",
            "value",
        ]
    )
    for tx in tx_list_sorted:
        df_tx.loc[len(df_tx)] = tx

    df_tx.to_pickle(f"./tx_{len(df_tx)}.pkl")
