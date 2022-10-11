"""
Only for test
Use .ipynb
"""

import pandas as pd
import numpy as np
import bitcoin
import datetime
from ping3 import ping
import random
import json
import re
import multiprocessing as mp

def gen_tx(url_list, user_list):
    send_ts=datetime.datetime.now()
    
    latency=None
    while not latency:
        url=random.choice(url_list)
        latency=ping(url)
    
    recv_ts=send_ts+datetime.timedelta(seconds=latency)
    
    from_user=random.choice(user_list)
    to_user=random.choice(user_list)
    value=random.random()*10
    
    label=0
    return [send_ts, recv_ts, latency, label, from_user, to_user, value]

if __name__ == "__main__":
    with open("./website.json", "r") as f:
        website_list=json.load(f)
        
    pattern=re.compile("http://(.+?)/")
    url_list=[re.findall(pattern, dic["home"])[0] for dic in website_list]
    
    df_user=pd.read_csv("./user.csv")
    user_list=df_user["address"].values
    
    core=8
    pool=mp.get_context("fork").Pool(core)
    tx_list=[]
    for _ in range(200):
        res=pool.apply_async(gen_tx, (url_list, user_list))
        tx_list.append(res)
        
    pool.close()
    pool.join()
    
    # jobs=[]
    # for _ in range(100):
    #     for i in range(core):
    #         proc=mp.Process(target=gen_tx, args=(url_list, user_list))
    #         jobs.append(proc)
            
    # for j in jobs:
    #     j.start()
    # for j in jobs:
    #     j.join()

    print(len(tx_list))
    print(tx_list[:10])
    